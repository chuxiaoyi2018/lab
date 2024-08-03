/*****************************************************************************
 *
 *    Cpp Reference
 *    ref:https://en.cppreference.com/w/cpp/utility/optional
 *
 *****************************************************************************/

#include<iostream>
#include<optional>


#include <iostream>
#include <mutex>
#include <string_view>
#include <syncstream>
#include <thread>

//===------------------------------------------------------------===//
// Case 1
//===------------------------------------------------------------===//
volatile int g_i = 0;
std::mutex g_i_mutex;  // protects g_i
 
void safe_increment(int iterations)
{
    // const std::lock_guard<std::mutex> gurad(g_i_mutex);
    const std::lock_guard<std::mutex> lock(g_i_mutex);
    while (iterations-- > 0)
        g_i = g_i + 1;
    std::cout << "thread #" << std::this_thread::get_id() << ", g_i: " << g_i << '\n';
 
    // g_i_mutex is automatically released when lock goes out of scope
}
 
void unsafe_increment(int iterations)
{
    while (iterations-- > 0)
        g_i = g_i + 1;
    std::osyncstream(std::cout) << "thread #" << std::this_thread::get_id()
                                << ", g_i: " << g_i << '\n';
}
 
int case_1()
{
    auto test = [](std::string_view fun_name, auto fun)
    {
        g_i = 0;
        std::cout << fun_name << ":\nbefore, g_i: " << g_i << '\n';
        {
            std::jthread t1(fun, 1'000'000);
            std::jthread t2(fun, 1'000'000);
        }
        std::cout << "after, g_i: " << g_i << "\n\n";
    };
    test("safe_increment", safe_increment);
    test("unsafe_increment", unsafe_increment);
    return 0;
}

//===------------------------------------------------------------===//
// Case 2
//===------------------------------------------------------------===//
template<typename T> class my_lock_guard {
public:
    // 在 std::mutex 的定义中，下面两个函数被删除了
    // mutex(const mutex&) = delete;
    // mutex& operator=(const mutex&) = delete;
    // 因此这里必须传递引用
    void note(auto s) { std::cout << "  " << s << " #" << '\n'; }

    my_lock_guard(T& mutex) :mutex_(mutex){
        // 构造加锁
        mutex_.lock();
        note("mutex lock");
    }

    ~my_lock_guard() {
        // 析构解锁
        mutex_.unlock();
        note("mutex unlock");
    }
private:
    // 不可赋值，不可拷贝
    my_lock_guard(my_lock_guard const&);
    my_lock_guard& operator=(my_lock_guard const&);
private:
    T& mutex_;
};

int shared_data = 0;
std::mutex shared_data_mutex;

void safe_increment_shared_data() {
    my_lock_guard<std::mutex> lock(shared_data_mutex);
    ++shared_data;
}

int case_2() {
    // if lock there, will to wait forever
    // my_lock_guard<std::mutex> lock(shared_data_mutex);
    const int num_threads = 5;
    std::thread t[num_threads];
    shared_data = 0;

    for (int i = 0; i < num_threads; ++i) {
        t[i] = std::thread(safe_increment_shared_data);
    }

    for (int i = 0; i < num_threads; ++i) {
        t[i].join();
    }

    std::cout << "Final shared data value: " << shared_data << std::endl;
    return 0;
}


int main() {
    // case 1
    case_1();

    // case 2
    case_2();
}