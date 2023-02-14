#include <sycl/sycl.hpp>
#include <iostream>
using namespace  cl::sycl;

int main(int argc, char **argv) {

	std::cout << "Starting" << std::endl;
	sycl::queue Q(sycl::cpu_selector{});

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;

	Q.submit([&](handler &cgh) {
		// Create a output stream
		sycl::stream sout(1024, 256, cgh);
		// Submit a unique task, using a lambda
		cgh.single_task([=]() {
			sout << "Hello, World!" << sycl::endl;
		}); // End of the kernel function
	});   // End of the queue commands. The kernel is now submited

	// wait for all queue submissions to complete
	Q.wait();


  return 0;
}