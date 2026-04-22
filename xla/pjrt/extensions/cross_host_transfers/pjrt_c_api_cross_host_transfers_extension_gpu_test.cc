/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/debug_options_flags.h"
#include "xla/ffi/api/ffi.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/extensions/cross_host_transfers/pjrt_c_api_cross_host_transfers_extension.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla_data.pb.h"

namespace pjrt {
namespace {

static std::string SuccessfulCrossHostSendReceiveTestName(
    const ::testing::TestParamInfo<int>& info) {
  return absl::StrFormat("num_arrays_%d", info.param);
}

struct SuccessfulCrossHostTransferTestParam {
  int num_rank_0_to_rank_1;
  int num_rank_1_to_rank_0;
};

static std::string SuccessfulCrossHostTransferTestName(
    const ::testing::TestParamInfo<SuccessfulCrossHostTransferTestParam>&
        info) {
  return absl::StrFormat("num_rank_0_to_rank_1_%d_num_rank_1_to_rank_0_%d",
                         info.param.num_rank_0_to_rank_1,
                         info.param.num_rank_1_to_rank_0);
}

absl::StatusOr<PJRT_Client_Create_Args> BuildCreateArg(
    ::pjrt::PJRT_KeyValueCallbackData* kv_callback_data,
    const std::vector<PJRT_NamedValue>& c_options) {
  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.create_options = c_options.data();
  args.num_options = c_options.size();
  args.kv_get_callback = kv_callback_data->c_kv_get;
  args.kv_get_user_arg = &kv_callback_data->kv_get_c_func;
  args.kv_put_callback = kv_callback_data->c_kv_put;
  args.kv_put_user_arg = &kv_callback_data->kv_put_c_func;
  args.kv_try_get_user_arg = &kv_callback_data->kv_try_get_c_func;
  args.kv_try_get_callback = kv_callback_data->c_kv_try_get;
  args.client = nullptr;
  return args;
}

absl::Span<PJRT_Device* const> GetClientAddressableDevices(
    PJRT_Client* client, const PJRT_Api* api) {
  PJRT_Client_AddressableDevices_Args addr_args;
  addr_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  addr_args.extension_start = nullptr;
  addr_args.client = client;
  PJRT_Error* error = api->PJRT_Client_AddressableDevices(&addr_args);
  CHECK(error == nullptr);
  return absl::MakeSpan(addr_args.addressable_devices,
                        addr_args.num_addressable_devices);
}

PJRT_RawBuffer* CreateRawAliasOfBuffer(const PJRT_Api* api,
                                       PJRT_RawBuffer_Extension* extension,
                                       PJRT_Buffer* buffer) {
  PJRT_RawBuffer_CreateRawAliasOfBuffer_Args args;
  args.struct_size = PJRT_RawBuffer_CreateRawAliasOfBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  args.raw_buffer = nullptr;
  auto error = std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter>(
      extension->PJRT_RawBuffer_CreateRawAliasOfBuffer(&args),
      ::pjrt::MakeErrorDeleter(api));
  CHECK_EQ(error, nullptr);
  return args.raw_buffer;
}

void DestroyRawBuffer(const PJRT_Api* api, PJRT_RawBuffer_Extension* extension,
                      PJRT_RawBuffer* buffer) {
  PJRT_RawBuffer_Destroy_Args args;
  args.struct_size = PJRT_RawBuffer_Destroy_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  auto error = std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter>(
      extension->PJRT_RawBuffer_Destroy(&args), ::pjrt::MakeErrorDeleter(api));
  CHECK_EQ(error, nullptr);
}

// Helper struct and function to perform basic initialization for all cross host
// transfer tests (distributed runtime service creation, KV store creation, PJRT
// client creation, etc).
struct PreparedCrossHostTransferTest {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  std::shared_ptr<::pjrt::PJRT_KeyValueCallbackData> kv_callback_data;
  std::unique_ptr<PJRT_Client, ::pjrt::PJRT_ClientDeleter> client;
  const PJRT_Api* api;
  PJRT_CrossHostTransfers_Extension* cross_host_transfers_extension;
  PJRT_RawBuffer_Extension* raw_buffer_extension;
};

absl::StatusOr<PreparedCrossHostTransferTest> PrepareCrossHostTransferTest(
    int rank_id, absl::string_view log_prefix) {
  PreparedCrossHostTransferTest prepared_test;

  // Rank 0 creates a coordination service on so both processes can find each
  // other via the distributed runtime (port chosen arbitrarily).
  if (rank_id == 0) {
    LOG(INFO) << log_prefix << ": creating coordination service";
    TF_ASSIGN_OR_RETURN(
        prepared_test.service,
        xla::GetDistributedRuntimeService(
            "127.0.0.1:12347",
            xla::CoordinationServiceImpl::Options{/*num_nodes=*/2}));
    LOG(INFO) << log_prefix << ": created service";
  }

  // Connect to the coordination service.
  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = rank_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12347", distributed_options);
  LOG(INFO) << log_prefix << ": connecting distributed client";
  TF_QCHECK_OK(distributed_client->Connect());
  LOG(INFO) << log_prefix << ": distributed client connected";

  auto kv_store = xla::GetDistributedKeyValueStore(distributed_client, "foo");
  prepared_test.kv_callback_data =
      ::pjrt::ConvertToCKeyValueCallbacks(kv_store);
  xla::ClientLibrary::DestroyLocalInstances();

  prepared_test.api = GetPjrtApi();
  prepared_test.cross_host_transfers_extension =
      pjrt::FindExtension<PJRT_CrossHostTransfers_Extension>(
          prepared_test.api,
          PJRT_Extension_Type::PJRT_Extension_Type_CrossHostTransfers);
  CHECK_NE(prepared_test.cross_host_transfers_extension, nullptr);

  prepared_test.raw_buffer_extension =
      pjrt::FindExtension<PJRT_RawBuffer_Extension>(
          prepared_test.api,
          PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);

  // Create the GPU client.
  absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
      {"num_nodes", static_cast<int64_t>(2)},
      {"node_id", static_cast<int64_t>(rank_id)},
      {"visible_devices", std::vector<int64_t>({rank_id})}};
  TF_ASSIGN_OR_RETURN(std::vector<PJRT_NamedValue> c_options,
                      ::pjrt::ConvertToPjRtNamedValueList(options));
  TF_ASSIGN_OR_RETURN(
      PJRT_Client_Create_Args create_arg,
      BuildCreateArg(prepared_test.kv_callback_data.get(), c_options));
  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> error(
      prepared_test.api->PJRT_Client_Create(&create_arg),
      ::pjrt::MakeErrorDeleter(prepared_test.api));
  if (error != nullptr) {
    return error->status;
  }
  prepared_test.client =
      std::unique_ptr<PJRT_Client, ::pjrt::PJRT_ClientDeleter>(
          create_arg.client, ::pjrt::MakeClientDeleter(prepared_test.api));

  return prepared_test;
}

// SuccessfulCrossHostSendReceiveTest tests CrossHost{Send/Receive}Buffers.
class SuccessfulCrossHostSendReceiveTest
    : public ::testing::TestWithParam<int> {};

TEST_P(SuccessfulCrossHostSendReceiveTest, SuccessfulCrossHostSendReceive) {
  int num_arrays = GetParam();

  tsl::SubProcess sender;
  tsl::SubProcess receiver;
  absl::string_view log_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");

  std::vector<std::string> sender_argv;
  sender_argv.push_back("successful_cross_host_transfer_test");
  sender_argv.push_back("--test_to_run=SuccessfulCrossHostSendReceiveHelper");
  sender_argv.push_back("--cross_host_test_role=sender");
  sender_argv.push_back(absl::StrFormat("--num_arrays=%d", num_arrays));
  sender_argv.push_back(absl::StrFormat("--log_dir=%s", log_dir));

  std::vector<std::string> receiver_argv;
  receiver_argv.push_back("successful_cross_host_transfer_test");
  receiver_argv.push_back("--test_to_run=SuccessfulCrossHostSendReceiveHelper");
  receiver_argv.push_back("--cross_host_test_role=receiver");
  receiver_argv.push_back(absl::StrFormat("--num_arrays=%d", num_arrays));
  receiver_argv.push_back(absl::StrFormat("--log_dir=%s", log_dir));

  sender.SetProgram("/proc/self/exe", sender_argv);
  sender.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  sender.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  receiver.SetProgram("/proc/self/exe", receiver_argv);
  receiver.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  receiver.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  ASSERT_TRUE(receiver.Start());
  ASSERT_TRUE(sender.Start());

  std::string sender_stdout, sender_stderr;
  std::string receiver_stdout, receiver_stderr;

  int sender_status =
      sender.Communicate(nullptr, &sender_stdout, &sender_stderr);
  int receiver_status =
      receiver.Communicate(nullptr, &receiver_stdout, &receiver_stderr);

  EXPECT_EQ(sender_status, 0) << "sender stdout:\n"
                              << sender_stdout << "\nsender stderr:\n"
                              << sender_stderr;
  EXPECT_EQ(receiver_status, 0) << "receiver stdout:\n"
                                << receiver_stdout << "\nreceiver stderr:\n"
                                << receiver_stderr;
}

INSTANTIATE_TEST_SUITE_P(SuccessfulCrossHostSendReceive,
                         SuccessfulCrossHostSendReceiveTest,
                         ::testing::ValuesIn({1, 2, 3}),
                         SuccessfulCrossHostSendReceiveTestName);

absl::Status SuccessfulCrossHostSendReceiveTestBody(bool is_sender,
                                                    int num_arrays) {
  std::string log_prefix = is_sender ? "sender" : "receiver";

  TF_ASSIGN_OR_RETURN(
      PreparedCrossHostTransferTest prepared_test,
      PrepareCrossHostTransferTest(is_sender ? 0 : 1, log_prefix));

  auto api = prepared_test.api;
  PJRT_CrossHostTransfers_Extension* cross_host_transfers_extension =
      prepared_test.cross_host_transfers_extension;
  PJRT_Client* client = prepared_test.client.get();
  CHECK_NE(cross_host_transfers_extension
               ->PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice,
           nullptr);

  std::vector<int64_t> shape = {2, 3};
  xla::Shape xla_shape =
      xla::ShapeUtil::MakeShape(xla::F32, /*dimensions=*/shape);

  // Sender logic.
  if (is_sender) {
    std::vector<PJRT_Buffer*> raw_buffers;
    std::vector<xla::GlobalDeviceId> dst_device_ids;
    std::vector<xla::CrossHostTransferKey> transfer_keys;
    raw_buffers.reserve(num_arrays);
    dst_device_ids.reserve(num_arrays);
    transfer_keys.reserve(num_arrays);
    for (int i = 0; i < num_arrays; ++i) {
      // Create buffers to send.
      std::vector<float> data = {1, 2, 3, 4, 5, 6 * static_cast<float>(i)};
      PJRT_Client_BufferFromHostBuffer_Args args;
      args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
      args.extension_start = nullptr;
      args.data = data.data();
      args.type = ::pjrt::ConvertToPjRtBufferType(xla_shape.element_type());
      args.dims = xla_shape.dimensions().data();
      args.num_dims = xla_shape.dimensions().size();
      args.byte_strides = nullptr;
      args.num_byte_strides = 0;
      args.device_layout = nullptr;
      args.host_buffer_semantics = ::pjrt::ConvertToPjRtHostBufferSemantics(
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall);
      args.client = client;
      args.device = GetClientAddressableDevices(client, api)[0];
      args.memory = nullptr;

      auto transfer_error =
          std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter>{
              api->PJRT_Client_BufferFromHostBuffer(&args),
              ::pjrt::MakeErrorDeleter(api)};
      if (transfer_error != nullptr) {
        return transfer_error->status;
      }
      CHECK_OK(args.buffer->buffer->GetReadyFuture().Await());
      std::unique_ptr<PJRT_Event, PJRT_EventDeleter> event(
          args.done_with_host_buffer, MakeEventDeleter(api));

      raw_buffers.push_back(args.buffer);
      CHECK_OK(event->future.Await());
      xla::GlobalDeviceId src_device_id =
          args.device->device->global_device_id();
      dst_device_ids.push_back(1 - src_device_id);
      transfer_keys.push_back(xla::CrossHostTransferKey(i));
    };

    // Send the list of buffers.
    PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args send_args;
    send_args.struct_size =
        PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args_STRUCT_SIZE;
    send_args.extension_start = nullptr;
    send_args.client = client;
    send_args.num_buffers = raw_buffers.size();
    send_args.buffers = raw_buffers.data();
    send_args.dst_global_device_ids = dst_device_ids.data();
    send_args.transfer_keys = transfer_keys.data();
    std::vector<PJRT_Event*> temp_events(raw_buffers.size());
    send_args.send_events = temp_events.data();
    cross_host_transfers_extension
        ->PJRT_Transfers_PJRT_Client_CrossHostSendBuffers(&send_args);

    for (int i = 0; i < num_arrays; ++i) {
      CHECK_OK(send_args.send_events[i]->future.Await());
      std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
          raw_buffers[i], ::pjrt::MakeBufferDeleter(api));
      std::unique_ptr<PJRT_Event, PJRT_EventDeleter> send_event(
          send_args.send_events[i], MakeEventDeleter(api));
      CHECK_OK(send_event->future.Await());
    }
  } else {
    // Receive some data.
    std::vector<xla::Literal> expected_literals;
    expected_literals.reserve(num_arrays);
    for (int i = 0; i < num_arrays; ++i) {
      expected_literals.push_back(xla::LiteralUtil::CreateR2<float>(
          {{1, 2, 3}, {4, 5, 6 * static_cast<float>(i)}}));
    }
    std::vector<xla::Shape> shapes;
    std::vector<xla::GlobalDeviceId> src_device_ids;
    std::vector<xla::CrossHostTransferKey> transfer_keys;
    std::vector<size_t> shape_num_dims;
    std::vector<const int64_t*> num_dims;
    std::vector<PJRT_Buffer_Type> element_types;
    std::vector<PJRT_Buffer_MemoryLayout*> layouts;
    shapes.reserve(num_arrays);
    src_device_ids.reserve(num_arrays);
    transfer_keys.reserve(num_arrays);
    shape_num_dims.reserve(num_arrays);
    num_dims.reserve(num_arrays);
    element_types.reserve(num_arrays);
    layouts.reserve(num_arrays);
    xla::GlobalDeviceId dst_device_id =
        GetClientAddressableDevices(client, api)[0]->device->global_device_id();
    for (int i = 0; i < num_arrays; ++i) {
      shapes.push_back(xla_shape);
      src_device_ids.push_back(xla::GlobalDeviceId(1 - dst_device_id));
      transfer_keys.push_back(xla::CrossHostTransferKey(i));
      shape_num_dims.push_back(shapes.back().dimensions().size());
      num_dims.push_back(shapes.back().dimensions().data());
      element_types.push_back(
          ::pjrt::ConvertToPjRtBufferType(shapes.back().element_type()));
      layouts.push_back(nullptr);
    }

    PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args recv_args;
    recv_args.struct_size =
        PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args_STRUCT_SIZE;
    recv_args.extension_start = nullptr;
    recv_args.client = client;
    recv_args.num_shapes = shapes.size();
    recv_args.shape_num_dims = shape_num_dims.data();
    recv_args.num_dims = num_dims.data();
    recv_args.element_types = element_types.data();
    recv_args.layouts = layouts.data();
    recv_args.device = GetClientAddressableDevices(client, api)[0];
    recv_args.src_global_device_ids = src_device_ids.data();
    recv_args.transfer_keys = transfer_keys.data();
    std::vector<PJRT_Buffer*> temp_buffers(shapes.size());
    recv_args.buffers = temp_buffers.data();
    cross_host_transfers_extension
        ->PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers(&recv_args);

    for (int i = 0; i < num_arrays; ++i) {
      TF_RETURN_IF_ERROR(
          recv_args.buffers[i]->buffer->GetReadyFuture().Await());
      TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> recv_literal,
                          recv_args.buffers[i]->buffer->ToLiteral().Await());

      TF_RET_CHECK(
          xla::LiteralTestUtil::Equal(expected_literals[i], *recv_literal));
      std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
          recv_args.buffers[i], ::pjrt::MakeBufferDeleter(api));
    }
  }
  return absl::OkStatus();
}

// SuccessfulCrossHostTransferTest tests CrossHostTransferBuffers.
class SuccessfulCrossHostTransferTest
    : public ::testing::TestWithParam<SuccessfulCrossHostTransferTestParam> {};

TEST_P(SuccessfulCrossHostTransferTest, SuccessfulCrossHostTransfer) {
  SuccessfulCrossHostTransferTestParam param = GetParam();

  tsl::SubProcess sender;
  tsl::SubProcess receiver;
  absl::string_view log_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");

  std::vector<std::string> sender_argv;
  sender_argv.push_back("successful_cross_host_transfer_test");
  sender_argv.push_back("--test_to_run=SuccessfulCrossHostTransferHelper");
  sender_argv.push_back("--cross_host_transfer_test_rank=0");
  sender_argv.push_back(
      absl::StrFormat("--num_rank_0_to_rank_1=%d", param.num_rank_0_to_rank_1));
  sender_argv.push_back(
      absl::StrFormat("--num_rank_1_to_rank_0=%d", param.num_rank_1_to_rank_0));
  sender_argv.push_back(absl::StrFormat("--log_dir=%s", log_dir));

  std::vector<std::string> receiver_argv;
  receiver_argv.push_back("successful_cross_host_transfer_test");
  receiver_argv.push_back("--test_to_run=SuccessfulCrossHostTransferHelper");
  receiver_argv.push_back("--cross_host_transfer_test_rank=1");
  receiver_argv.push_back(
      absl::StrFormat("--num_rank_0_to_rank_1=%d", param.num_rank_0_to_rank_1));
  receiver_argv.push_back(
      absl::StrFormat("--num_rank_1_to_rank_0=%d", param.num_rank_1_to_rank_0));
  receiver_argv.push_back(absl::StrFormat("--log_dir=%s", log_dir));

  sender.SetProgram("/proc/self/exe", sender_argv);
  sender.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  sender.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  receiver.SetProgram("/proc/self/exe", receiver_argv);
  receiver.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  receiver.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  ASSERT_TRUE(receiver.Start());
  ASSERT_TRUE(sender.Start());

  std::string sender_stdout, sender_stderr;
  std::string receiver_stdout, receiver_stderr;

  int sender_status =
      sender.Communicate(nullptr, &sender_stdout, &sender_stderr);
  int receiver_status =
      receiver.Communicate(nullptr, &receiver_stdout, &receiver_stderr);

  EXPECT_EQ(sender_status, 0) << "sender stdout:\n"
                              << sender_stdout << "\nsender stderr:\n"
                              << sender_stderr;
  EXPECT_EQ(receiver_status, 0) << "receiver stdout:\n"
                                << receiver_stdout << "\nreceiver stderr:\n"
                                << receiver_stderr;
}

INSTANTIATE_TEST_SUITE_P(
    SuccessfulCrossHostTransfer, SuccessfulCrossHostTransferTest,
    ::testing::ValuesIn(std::vector<SuccessfulCrossHostTransferTestParam>{
        {1, 0}, {1, 1}, {2, 1}}),
    SuccessfulCrossHostTransferTestName);

absl::Status SuccessfulCrossHostTransferTestBody(int rank_id,
                                                 int num_rank_0_to_rank_1,
                                                 int num_rank_1_to_rank_0) {
  std::string log_prefix = absl::StrFormat("rank %d", rank_id);
  int num_transfers = num_rank_0_to_rank_1 + num_rank_1_to_rank_0;

  TF_ASSIGN_OR_RETURN(PreparedCrossHostTransferTest prepared_test,
                      PrepareCrossHostTransferTest(rank_id, log_prefix));

  auto api = prepared_test.api;
  PJRT_CrossHostTransfers_Extension* cross_host_transfers_extension =
      prepared_test.cross_host_transfers_extension;
  PJRT_RawBuffer_Extension* raw_buffer_extension =
      prepared_test.raw_buffer_extension;
  PJRT_Client* client = prepared_test.client.get();
  CHECK_NE(cross_host_transfers_extension
               ->PJRT_Transfers_PJRT_Client_CrossHostTransferBuffers,
           nullptr);
  CHECK_NE(raw_buffer_extension, nullptr);
  CHECK_NE(raw_buffer_extension->PJRT_RawBuffer_CreateRawAliasOfBuffer,
           nullptr);
  CHECK_NE(raw_buffer_extension->PJRT_RawBuffer_Destroy, nullptr);

  // Prepare the data sent for each transfer.
  // rank_id 0 sends buffers with data:
  //  [0, ..., 255]
  //  [1000, ..., 1255]
  //  [2000, ..., 2255]
  //  ...
  // rank_id 1 sends buffers with data:
  //  [10_000, ..., 10_255]
  //  [11_000, ..., 11_255]
  //  [12_000, ..., 12_255]
  //  ...
  std::vector<std::vector<int32_t>> transferred_data;
  transferred_data.reserve(num_transfers);
  for (int i = 0; i < num_rank_0_to_rank_1; ++i) {
    std::vector<int32_t> curr_data(256);
    absl::c_iota(curr_data, 1000 * i);
    transferred_data.push_back(std::move(curr_data));
  }
  for (int i = 0; i < num_rank_1_to_rank_0; ++i) {
    std::vector<int32_t> curr_data(256);
    absl::c_iota(curr_data, 10000 + 1000 * i);
    transferred_data.push_back(std::move(curr_data));
  }
  xla::Shape transfer_shape =
      xla::ShapeUtil::MakeShape(xla::S32, std::vector<int64_t>{256});

  // Initial values that will be populated in receive buffers (all zeros).
  std::vector<int32_t> initial_zero_values(256, 0);

  // The send / receive PjRtBuffers this rank allocates.
  std::vector<std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter>>
      owned_buffers;
  owned_buffers.reserve(num_transfers);
  std::vector<PJRT_RawBuffer*> raw_buffers;
  raw_buffers.reserve(num_transfers);
  std::vector<xla::GlobalDeviceId> src_global_device_ids;
  src_global_device_ids.reserve(num_transfers);
  std::vector<xla::GlobalDeviceId> dst_global_device_ids;
  dst_global_device_ids.reserve(num_transfers);

  LOG(INFO) << log_prefix << ": preparing transfers.";
  for (int i = 0; i < num_transfers; ++i) {
    int src_global_device_id = i < num_rank_0_to_rank_1 ? 0 : 1;
    int dst_global_device_id = i < num_rank_0_to_rank_1 ? 1 : 0;
    bool is_sender = rank_id == src_global_device_id;

    PJRT_Client_BufferFromHostBuffer_Args args;
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.data =
        is_sender ? transferred_data[i].data() : initial_zero_values.data();
    args.type = ::pjrt::ConvertToPjRtBufferType(transfer_shape.element_type());
    args.dims = transfer_shape.dimensions().data();
    args.num_dims = transfer_shape.dimensions().size();
    args.byte_strides = nullptr;
    args.num_byte_strides = 0;
    args.device_layout = nullptr;
    args.host_buffer_semantics = ::pjrt::ConvertToPjRtHostBufferSemantics(
        xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall);
    args.client = client;
    args.device = GetClientAddressableDevices(client, api)[0];
    args.memory = nullptr;  // Uses the device's default memory space.

    auto transfer_error =
        std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter>{
            api->PJRT_Client_BufferFromHostBuffer(&args),
            ::pjrt::MakeErrorDeleter(api)};
    if (transfer_error != nullptr) {
      return transfer_error->status;
    }
    CHECK_OK(args.buffer->buffer->GetReadyFuture().Await());
    std::unique_ptr<PJRT_Event, PJRT_EventDeleter> event(
        args.done_with_host_buffer, MakeEventDeleter(api));
    CHECK_OK(event->future.Await());

    raw_buffers.push_back(
        CreateRawAliasOfBuffer(api, raw_buffer_extension, args.buffer));
    owned_buffers.emplace_back(args.buffer, ::pjrt::MakeBufferDeleter(api));
    src_global_device_ids.push_back(xla::GlobalDeviceId(src_global_device_id));
    dst_global_device_ids.push_back(xla::GlobalDeviceId(dst_global_device_id));
  }

  // Perform transfers.
  LOG(INFO) << log_prefix << ": enqueuing transfers";
  PJRT_Transfers_PJRT_Client_CrossHostTransferBuffers_Args transfer_args;
  transfer_args.struct_size =
      PJRT_Transfers_PJRT_Client_CrossHostTransferBuffers_Args_STRUCT_SIZE;
  transfer_args.extension_start = nullptr;
  transfer_args.client = client;
  // We pass in zero transfer dependencies because when creating the
  // send/receive buffers, we blocked until they were ready.
  // TODO(asrao): Once PJRT C API methods to decompose a PJRT_Buffer into a
  // PJRT_RawBuffer + PJRT_DeviceEvent are exposed, add a test to the C API that
  // includes some transfer dependencies.
  transfer_args.num_dependencies = 0;
  transfer_args.transfer_dependencies = nullptr;
  transfer_args.num_transfers = raw_buffers.size();
  transfer_args.src_global_device_ids = src_global_device_ids.data();
  transfer_args.dst_global_device_ids = dst_global_device_ids.data();
  transfer_args.raw_buffers = raw_buffers.data();
  std::vector<PJRT_DeviceEvent*> transfer_events(raw_buffers.size());
  transfer_args.transfer_events = transfer_events.data();
  auto transfer_error = std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter>{
      cross_host_transfers_extension
          ->PJRT_Transfers_PJRT_Client_CrossHostTransferBuffers(&transfer_args),
      ::pjrt::MakeErrorDeleter(api)};
  if (transfer_error != nullptr) {
    return transfer_error->status;
  }

  // We explicitly block on the transfer events here while retaining references
  // to the PJRT_Buffers, so there is no danger of them being deallocated while
  // the transfer is in flight. Because of this, we do not register a usage
  // event for the transfer.
  // TODO(asrao): Once C APIs for adding usage PJRT_DeviceEvents to PJRT_Buffers
  // are available, register these as usage events.
  for (int i = 0; i < transfer_events.size(); ++i) {
    xla::PjRtDeviceEventRef transfer_event(*transfer_events[i]);
    tsl::BlockUntilReady(transfer_event.async_value());
    delete transfer_events[i];
  }

  for (int i = 0; i < num_transfers; ++i) {
    TF_RETURN_IF_ERROR(owned_buffers[i]->buffer->GetReadyFuture().Await());
    TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> buffer_literal,
                        owned_buffers[i]->buffer->ToLiteral().Await());
    const std::vector<int32_t>& expected_data = transferred_data[i];
    auto expected_literal = xla::LiteralUtil::CreateR1<int32_t>(expected_data);
    TF_RET_CHECK(
        xla::LiteralTestUtil::Equal(expected_literal, *buffer_literal));
  }

  for (PJRT_RawBuffer* raw_buffer : raw_buffers) {
    DestroyRawBuffer(api, raw_buffer_extension, raw_buffer);
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace pjrt

int main(int argc, char* argv[]) {
  std::string test_to_run;
  std::string cross_host_test_role;
  int num_arrays = -1;
  int cross_host_transfer_test_rank = -1;
  int num_rank_0_to_rank_1 = -1;
  int num_rank_1_to_rank_0 = -1;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("test_to_run", &test_to_run,
                "Test parameter for the multiprocess helper to run."),
      tsl::Flag("cross_host_test_role", &cross_host_test_role,
                "Test parameter for cross host helpers; either 'sender' or "
                "'receiver'."),
      tsl::Flag("cross_host_transfer_test_rank", &cross_host_transfer_test_rank,
                "Test parameter for SuccessfulCrossHostTransferHelper; either "
                "0 or 1."),
      tsl::Flag("num_arrays", &num_arrays,
                "Test parameter for SuccessfulCrossHostSendReceiveHelper; "
                "number of arrays to transfer."),
      tsl::Flag("num_rank_0_to_rank_1", &num_rank_0_to_rank_1,
                "Test parameter for SuccessfulCrossHostTransferHelper; "
                "number of transfers from rank 0 to rank 1."),
      tsl::Flag("num_rank_1_to_rank_0", &num_rank_1_to_rank_0,
                "Test parameter for SuccessfulCrossHostTransferHelper; "
                "number of transfers from rank 1 to rank 0.")};

  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);

  testing::InitGoogleTest(&argc, argv);
  if (test_to_run.empty()) {
    return RUN_ALL_TESTS();
  }
  if (test_to_run == "SuccessfulCrossHostSendReceiveHelper") {
    if (cross_host_test_role == "sender") {
      return pjrt::SuccessfulCrossHostSendReceiveTestBody(/*is_sender=*/true,
                                                          num_arrays)
          .raw_code();
    }
    if (cross_host_test_role == "receiver") {
      return pjrt::SuccessfulCrossHostSendReceiveTestBody(/*is_sender=*/false,
                                                          num_arrays)
          .raw_code();
    }
    return static_cast<int>(absl::StatusCode::kInvalidArgument);
  }
  if (test_to_run == "SuccessfulCrossHostTransferHelper") {
    if (cross_host_transfer_test_rank != 0 &&
        cross_host_transfer_test_rank != 1) {
      return static_cast<int>(absl::StatusCode::kInvalidArgument);
    }
    return pjrt::SuccessfulCrossHostTransferTestBody(
               cross_host_transfer_test_rank, num_rank_0_to_rank_1,
               num_rank_1_to_rank_0)
        .raw_code();
  }
  return static_cast<int>(absl::StatusCode::kInvalidArgument);
}
