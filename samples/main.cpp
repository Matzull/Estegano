#include <sycl/sycl.hpp>
#include <iostream>

int main(int argc, char **argv) {

	std::cout << "Starting" << std::endl;
	sycl::queue Q{sycl::device{sycl::default_selector_v}};

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;

	Q.submit([&](sycl::handler &cgh) {
		// Create a output stream
		sycl::stream sout(1024, 256, cgh);
		// Submit a unique task, using a lambda
		cgh.parallel_for(sycl::range<1>(10), [=](sycl::id<1> item) {
			sout << "Hello, World!" << sycl::endl;
		}); // End of the kernel function
	}).wait();   // End of the queue commands. The kernel is now submited
	printf("Hey\n");
	// wait for all queue submissions to complete
	Q.wait();


  return 0;
}