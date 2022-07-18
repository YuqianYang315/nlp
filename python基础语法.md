# 教程
## 数学函数
abs(x)绝对值
ceil（x）:math.ceil(x)返回数字的上入整数
floor（x):math.floor(x)返回数字的下设整数
log(x):math.log(math.e)=1.0,math.log(100,10)=10
log10(x):math.log10（100）=2
max()
min()
modf(x)返回x的整数部分和小数部分，两个部分数值符号与x相同，整数部分以浮点型表示
pow（x,y）:x**y后的值
round(x,[n])返回浮点数的四舍五入值，如给出N,则代表摄入到小数点后的位数
sqrt（x）返回数字的平方根
## 随机函数
choice(seq):random.choice(range(10)),从0-9中随机挑选一个

randrange:random.randrange([start,]stop[,step]),如random.randrange(1,100,2)从1-100中选一个奇数，random.randrange(100)从0-99选取一个随机数

random():random.random()随机生成下一个实数，在【0，1）内

seed():random.seed()

shuffle(lst)：将序列的所有元素随机排序random.shuffle([1,2,3])

uniform(x,y):随机生成下一个实数，在【x，y】内

## 字符串
### python转义字符
\b 退格

\000 空

\n 换行

\v 纵向制表符

\t 横向制表符

\r 将\r后的内容穿衣到字符串开头，并注意替换开头部分的字符

\f换页
### python字符串格式化
%s 格式化字符串

%d 格式化整数

%f 格式化浮点数字，可指定小数点后的额精度
### 字符串内建函数
str.capitalize()
center(width,fillchar)

# 错误与异常
# assert
import sys
print(sys.platform)
assert ('linux' in sys.platform), '改代码只能在linux下运行'



# with
with open('/data/aif/personal/yuko/KDR_NLP/pdf_paragraph_text.csv',encoding='utf-8') as f :
    red=f.read()
#查看文件是否关闭
f.closed



# 面向对象
class Employee:
    '所有员工的基类'
    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print
        "Total Employee %d" % Employee.empCount

    def displayEmployee(self):
        print
        "Name : ", self.name, ", Salary: ", self.salary


"创建 Employee 类的第一个对象"
emp1 = Employee("Zara", 2000)
"创建 Employee 类的第二个对象"
emp2 = Employee("Manni", 5000)
emp1.displayEmployee()
emp2.displayEmployee()



## 通依以下函数访问属性
#检查是否存在一个属性
hasattr(emp1,'age')
getattr(emp1,'age')
setattr(emp1,'age',8)
delattr(emp1,'age')


Employee.__doc__
Employee.__base__
Employee.__module__
Employee.__dict__

# 查看是否可调用
callable(emp1)


# 命名空间、作用域

#Python 中只有模块（module），类（class）以及函数（def、lambda）才会引入新的作用域，其它的代码块（如 if/elif/else/、try/except、for/while等）是不会引入新的作用域的，也就是说这些语句内定义的变量，外部也可以访问

#在函数中修改全局变量的值
globals num

#修改嵌套作用域中对的变量（enclosing作用域）


def fun():
    num=1#嵌套作用域
    def fun_in():
        nu=2#局部作用域
nonlocal num


# 标准库

#操作系统接口
import os
os.getced()


# 内置库
import os
#返回当前工作目录
print(os.getcwd())



import glob

print(glob.glob('*.py'))

import sys
print(sys.argv)


# re




# random


print(random.choice(['apple','pear']))

print(random.sample(range(100),10))



print(random.random())
print(random.randrange(6))


# date
from datetime import date
now=date.today()
now.strftime('%m-%d-%y.%d%b%Y is a %A on the %d day of %B')
# '12-02-03. 02 Dec 2003 is a Tuesday on the 02 day of December.'
age=date.today()-date(1996,3,15)
age.days



# zlib
import zlib
s = b'witch which has which witches wrist watch'
t=zlib.compress(s)
len(t)
zlib.decompress(t)
#注意赋值的操作，解压和压缩源文件都不变


# 性能度量
from timeit import Timer
Timer('t=a;a=b','a=2;b=2').timeit()
Timer('t=a','a=2;b=2').timeit()

