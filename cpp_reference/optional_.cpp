/*****************************************************************************
 *
 *    Cpp Reference
 *    ref:https://en.cppreference.com/w/cpp/utility/optional
 *
 *****************************************************************************/

#include<iostream>
#include<optional>


int case_1(){
    std::optional<int> opt;
    int num = 0;

    // case 1
    std::cout << "please enter a number: ";
    std::cin >> num;

    opt = num;

    if(!opt.has_value()){
        std::cout << "the number you entered is empty" << std::endl;
    } else {
        std::cout << "the number you entered is " << opt.value() << std::endl;
    }
    return 0;
}

auto case_2(bool flag) {
    return flag ? std::optional<int>(1) : std::nullopt;
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
    auto opt = case_2(true);
    if(opt.has_value()){
        std::cout << "the number is " << opt.value() << std::endl;
    }

    opt = case_2(false);
    if(opt.has_value()){
        std::cout << "the number is " << opt.value() << std::endl;
    }

    // case 3
    std::optional<A> opt_a;

    opt_a.emplace("hello");

    opt_a.emplace("world");
    
    // case 4
    std::optional<A> opt_b;

    std::cout << "Assign:\n";
    opt_b = A("Lorem ipsum dolor sit amet, consectetur adipiscing elit nec.");
 
    std::cout << "Emplace:\n";
    // As opt contains a value it will also destroy that value
    opt_b.emplace("Lorem ipsum dolor sit amet, consectetur efficitur.");

    return 0;
}