#include "caffe/util/io.hpp"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: ./convert_net_proto_text_to_binary.bin"
      << " net_proto_text_file_in net_proto_binary_file_out" << std::endl;
    return -1;
  }
  caffe::NetParameter net_param;
  caffe::ReadProtoFromTextFileOrDie(argv[1], &net_param);
  caffe::WriteProtoToBinaryFile(net_param, argv[2]);
  return 0;
}
