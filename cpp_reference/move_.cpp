/*****************************************************************************
 *
 *    Cpp Reference
 *    ref:https://zh.cppreference.com/w/cpp/utility/move
 *    ref:https://wiki.sophgo.com/pages/viewpage.action?pageId=83314642
 *
 *****************************************************************************/


#include <iostream>
#include <utility>
#include <vector>
#include <string>

// v           str
//             Hello
// Hello(new)  Hello        (push_back)
// Hello                    (push_back(std::move))
int case_1()
{
    std::string str = "Hello";
    std::vector<std::string> v;

    v.push_back(str);
    std::cout << "After copy, str is \"" << str << "\"\n";

    v.push_back(std::move(str));
    std::cout << "After move, str is \"" << str << "\"\n";
    std::cout << "The contents of the vector are \"" << v[0]
                                         << "\", \"" << v[1] << "\"\n";
    return 0;
}

struct A
{
    // 定义了一个名为 A 的结构体，它有一个成员变量 s，类型为 std::string，以及一个静态成员变量 n 和一个非静态成员变量 id。
    std::string s;
 
    // A 的构造函数，它接受一个 std::string 参数。它使用 std::move 将字符串移动到成员变量 s，以避免不必要的复制。
    // id 成员被初始化为静态成员 n 的当前值，并且 n 被递增。构造函数调用 note 方法以打印一条消息表示对象已构造
    A(std::string str) : s(std::move(str)), id{n++} { note("+ constructed"); }

    // A 的析构函数，它调用 note 方法以打印一条消息表示对象已销毁。
    ~A() { note("~ destructed"); }

    // A 的拷贝构造函数，它接受一个 const A& 参数。
    A(const A& o) : s(o.s), id{n++} { note("+ copy constructed"); }

    // A 的移动构造函数，它接受一个右值引用参数，并通过 std::move 将 o 的字符串成员移动到新创建的对象中。这样可以避免复制字符串，而是重用现有的内存
    A(A&& o) : s(std::move(o.s)), id{n++} { note("+ move constructed"); }
 
    A& operator=(const A& other)
    {
        s = other.s;
        note("= copy assigned");
        return *this;
    }
 
    A& operator=(A&& other)
    {
        s = std::move(other.s);
        note("= move assigned");
        return *this;
    }
 
    inline static int n{};
    int id{};
    void note(auto s) { std::cout << "  " << s << " #" << id << '\n'; }
};

int main(){

    // case 1
    case_1();

    // case 2
    const A a("Hello");
    A b = a; // copy

    A a1("Hello");
    A c = std::move(a1); // move
    // A c1 = &c;         bad case
    A c1 = static_cast<A&&>(c); // good case

    // case 3
    

    return 0;
}