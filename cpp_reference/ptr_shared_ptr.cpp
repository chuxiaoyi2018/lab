/*****************************************************************************
 *
 *    Cpp Reference
 *    ref:https://zh.cppreference.com/w/cpp/utility/move
 *    ref:https://wiki.sophgo.com/pages/viewpage.action?pageId=93272534
 
 *    使用指针的主要目的是为了节省内存
 *    避免多次使用时候的拷贝
 *    但是使用指针会存在忘记释放内存的一个问题
 *
 *****************************************************************************/

#include <cstring>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

void read_memory() {
  FILE *fp = fopen("/proc/self/status", "r");
  char line[1000];
  while (fgets(line, 128, fp) != NULL) {
    std::cout << line;
  }
  fclose(fp);
}

std::vector<int> create_weight(int length) {
  std::vector<int> weight(length);
  for (int i = 0; i < length; i++) {
    weight[i] = i;
  }
  return weight;
}

int case_1() {
  std::vector<int> vec;
  vec = create_weight(1000000);
  read_memory();
  return 0;
}

int case_2() {
  std::shared_ptr<std::vector<int>> s_ptr;
  s_ptr = std::make_shared<std::vector<int>>(create_weight(1000000));
  read_memory();
  return 0;
}


int case_3() {
  std::vector<int>* ptr = new std::vector<int>(create_weight(1000000));
  read_memory();
  return 0;
}

struct Base {
    Base() {
        std::cout << "Base()" << std::endl;
    }
    ~Base() {
        std::cout << "~Base()" << std::endl;
    }
    int num = 0;
};

struct Derived : public Base {
    Derived() {
        std::cout << "Derived()" << std::endl;
    }
    ~Derived() {
        std::cout << "~Derived()" << std::endl;
    }
};

void print(auto rem, std::shared_ptr<Base> const &ptr) {
    std::cout << rem << "\n\tget() = " << ptr.get() << "\n\tuse_count() = "
    << ptr.use_count() << std::endl;
}


// copy shared_ptr when passing to function
// copy shared_ptr when create local pointer
void thr_method_1(std::shared_ptr<Base> p) {
    print("Local pointer in this thread :", p);
    std::this_thread::sleep_for(1s);
    std::shared_ptr<Base> lp = p;
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lock(io_mutex);
        print("Local pointer in this thread :", p);
    }
}


int case_4() {
    std::shared_ptr<Base> p1 = std::make_shared<Derived>();
    print("Create a shared Derived pointer (as a pointer to Base): ", p1);

    std::thread t1(thr_method_1, p1);
    print("Now use count: ", p1);
    t1.join(); // wait thread 1 to complete task
    // t2.join(); // wait thread 2 to complete task
    // t3.join();
    print("Now use count: ", p1);
    std::cout << "Destroying the shared pointer in main()" << std::endl;
    return 0;
}

// copy shared_ptr when passing to function
// copy shared_ptr when create local pointer
void thr_method_2(std::shared_ptr<Base> const &p) {
    print("Local pointer in this thread :", p);
    std::this_thread::sleep_for(1s);
    std::shared_ptr<Base> lp = p;
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lock(io_mutex);
        print("Local pointer in this thread :", p);
    }
}


int case_5() {
    std::shared_ptr<Base> p1 = std::make_shared<Derived>();
    print("Create a shared Derived pointer (as a pointer to Base): ", p1);

    std::thread t1(thr_method_2, p1);
    print("Now use count: ", p1);
    t1.join(); // wait thread 1 to complete task
    // t2.join(); // wait thread 2 to complete task
    // t3.join();
    print("Now use count: ", p1);
    std::cout << "Destroying the shared pointer in main()" << std::endl;
    return 0;
}

// copy shared_ptr when passing to function
// copy shared_ptr when create local pointer
// maybe not to find any method to avoid to create a new pointer in the function
void thr_method_3(std::shared_ptr<Base> const &p) {
    std::cout << "Num :" << p << std::endl;
    print("Local pointer in this thread :", p);
    std::this_thread::sleep_for(1s);
    std::shared_ptr<Base> lp = p;
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lock(io_mutex);
        print("Local pointer in this thread :", p);
    }
}


int case_6() {
    std::shared_ptr<Base> p1 = std::make_shared<Derived>();
    print("Create a shared Derived pointer (as a pointer to Base): ", p1);
    std::cout << "Num :" << p1->num << std::endl;

    std::thread t1(thr_method_3, std::move(p1)); // when std::move, use_count = 0, compiler empty the memory and num will be 0x56444ef0fec0
    // std::cout << "Num :" << p1->num << std::endl; // Segmentation fault
    print("Now use count: ", p1);
    t1.join(); // wait thread 1 to complete task
    // t2.join(); // wait thread 2 to complete task
    // t3.join();
    print("Now use count: ", p1);
    std::cout << "Destroying the shared pointer in main()" << std::endl;
    return 0;
}

int case_7() {
    std::shared_ptr<Base> p1 = std::make_shared<Derived>();
    print("Create a shared Derived pointer (as a pointer to Base): ", p1);
    std::shared_ptr<Base> p2 = p1;
    std::cout << "Num :" << p1->num << std::endl;
    print("Now use count: ", p1);
    print("Now use count: ", p2);

    std::thread t1(thr_method_3, std::move(p1)); // when std::move, use_count = 0, compiler empty the memory and num will be 0x56444ef0fec0
    // std::cout << "Num :" << p1->num << std::endl; // Segmentation fault
    print("Now use count: ", p1);
    t1.join(); // wait thread 1 to complete task
    // t2.join(); // wait thread 2 to complete task
    // t3.join();
    print("Now use count: ", p1);
    std::cout << "Destroying the shared pointer in main()" << std::endl;
    return 0;
}



int main() {

//   case_1();
//   case_2();
//   case_3();
//   case_4();
    
//   case_5();
//   case_6();
  case_7();
  return 0;
}