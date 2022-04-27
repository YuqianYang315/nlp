# def __getattr__(self, name)
 当取类中的函数不存在时，返回getattr中return的函数（值也赋给这个函数），若没有则报错

 eg:
 ```
 class A(object):
    def __init__(self, a, b):
        self.a1 = a
        self.b1 = b
        print('init')

    def mydefault(self, *args):
        print('default:' + str(args[0]))

    def __getattr__(self, name):
        print("other fn:", name)
        return self.mydefault#(33值也赋给mydefault)


a1 = A(10, 20)
a1.fn1(33)
a1.fn2('hello')
```

# Python classmethod 修饰符
classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等。
```
class A(object):
    bar = 1
    def func1(self):  
        print ('foo') 
    @classmethod
    def func2(cls):
        print ('func2')
        print (cls.bar)
        cls().func1()   # 调用 foo 方法
 
A.func2()               # 不需要实例化
```
输出：
```
func2
1
foo
```
