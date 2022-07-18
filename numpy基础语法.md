import numpy as np

# ndarry对象:N 维数组对象
a = np.array([1, 2], [2, 3])

a = np.array([1, 2, 3, 2, 2], ndimn=2)

a = np.array([1, 3, 2], dtype=complex)

# numpy数据类型
# numpy得数据类型是numpy.dtype的类实例
dt = np.dtype(np.int32)
# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
dt = np.dtype('i4')
# 首先创建结构化数据类型
# eg1
dt = np.dtype([('age', np.int8)])
a = np.array([(10,), (20,), (30,)], dtype=dt)
a['age']

# eg2
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
a = np.array([('abc', 21, 50), ('xyz', 18, 50)], dtype=dt)
print(a)
print(a['name'])

# 数组属性
ndarray.ndim
ndarray.shape
ndarray.reshape()
ndarray.itemsize

# numpy创建数组
np.empty([3, 2], dtype=int)
np.zeros(5)  # 默认浮点数
np.zeros((5,))
np.zeros((2, 2), dtype=int)
np.ones([2, 2], dtype=int)

np.frombuffer('hello world', dtype='S1')  # 用于实现动态数组，以流的形式读入转化为ndarray对象
np.fromiter(iter([1, 2]), dtype=float)  # 从可迭代对象中建立ndarray对象，返回一维数组

# 从数值范围创建数组
np.arange(5)
np.arange(10, 20, 2)
np.linspace()  # 创建一个等差数列
np.logspace()  # 创建一个等比数列

# numpy的切片和索引
a = np.arange(10)
a[slice(2, 7, 2)] == a[2:7:2]
# 省略号：
a = np.array([[2, 3, 4], [2, 3, 2], [7, 3, 3]])
a[..., 1]  # [3 3 3]
a[1, ...]  # [2,3,2]

# numpy高级索引
# NumPy 比一般的 Python 序列提供更多的索引方式。除了之前看到的用整数和切片的索引外，数组可以由整数数组索引、布尔索引及花式索引
# 整数数组索引
x = np.array([[1, 2], [3, 4], [5, 6]])
y = x[[0, 1, 2], [0, 1, 0]]  # 以下实例获取数组中(0,0)，(1,1)和(2,0)位置处的元素

y = x[np.array([0, 0], [3, 3]), np.array([0, 2], [0, 2])]
##以上实例获取数组中(0,0)，(0,2)和(3,0)(3,2)位置处的元素
# 布尔索引
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
y = x[(x > 5) & (x < 10)]
x[(x[..., 0] > 2) & (x[..., 1] < 3)]  # 可以针对某一列的数据，进而提取需要的数组。

# 花式索引

x = np.arange(32).reshape((8, 4))
y = x[[4, 2, 1, 7]]
#结果：
# [[16 17 18 19]
 # [ 8  9 10 11]
 # [ 4  5  6  7]
 # [28 29 30 31]]
y=x[np.ix_([1,5,7,2],[0,3,1,2])]
# 相当于y=np.array([[x[1,0], x[1,3], x[1,1], x[1,2]],\
#             [x[5,0], x[5,3], x[5,1],x[5,2]],\
#             [x[7,0] ,x[7,3], x[7,1], x[7,2]],\
#             [x[2,0], x[2,3], x[2,1], x[2,2]]])


# 广播（broadcast）
#广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式

# 迭代数组
for x in np.nditer(x,op_flags=['readwrite']):
    x[...]=2*x
np.copy(x,order='C')
#它拷贝的时候不是直接把对方的内存复制，而是按照上面 order 指定的顺序逐一拷贝。
# 其实他影响的就是迭代取值的时候，按照内存存储方式取值。不然表面上看两个一样（np.copy(x,order='F')与np.copy(x,order='C')）
# 迭代时，本身的存储顺序不会因为转置或 order = 'C' 或 'F' 而改变。


# 数组操作
#形状
np.transpose(x)==a.T
np.rollaxis(x,2,0)#将轴2滚到轴0
np.swapaxes(x,1,2)#一二轴交换
#维度


# np.



import numpy as np


a=np.arange(4).reshape(1,4)
np.broadcast_to(a,(4,4))
#拓展维度
a=np.expand_dims(a,0)
#删除一味的条目
np.squeeze(a,0)

# 连接数组
np.concatenate()

# 分割数组
a=np.arange(9)
print ('将数组分为三个大小相等的子数组：')
np.split(a,3)
print('左开右闭[array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8])]')
np.split(a,[4,7])

a=np.arange(4).reshape(4,4)
np.split(a,2,0)
#按第一维分割两个部分
# [array([[0, 1, 2, 3],
#        [4, 5, 6, 7]]), array([[ 8,  9, 10, 11],
#        [12, 13, 14, 15]])]
np.split(a,2,1)#按第二维分割两个部分
# [array([[ 0,  1],
#        [ 4,  5],
#        [ 8,  9],
#        [12, 13]]), array([[ 2,  3],
#        [ 6,  7],
#        [10, 11],
       # [14, 15]])]


# 数组中的添加与删除
a=np.arange(4)
#np.reshape只改变数据形状，np.resize返回指定形状的新数组
np.resize(a,(4,4))==np.broadcast_to(a,(4,4))

# np.append
numpy.append(arr, values, axis=None)


# np.insert
numpy.insert(arr, obj, values, axis)
如果值的类型转换为要插入，则它与输入数组不同。 插入没有原地的，函数会返回一个新数组。 此外，如果未提供轴，则输入数组会被展开。
# np.delete
numpy.delete 函数返回从输入数组中删除指定子数组的新数组。 与 insert() 函数的情况一样，如果未提供轴参数，则输入数组将展开。
Numpy.delete(arr, obj, axis)
# np.unique
numpy.unique(arr, return_index, return_inverse, return_counts)
a = np.array([5,2,6,2,7,5,6,8,2,9])
u,indices = np.unique(a, return_index = True)
index:'去重数组的索引数组：'
u:第一个数组的去重值：



# np数学函数
np.ceil(a)
#可以对整个数组进行操作，蛮好用的！

# 算数函数：加减乘除
add(),subtract(),multiply(),divide()
需要注意的是数组必须具有相同的形状或符合数组广播规则

#倒数
np.reciprocal(a)
np.pow(a,b)
np.mod(a,b)



# 统计函数

#统计最小/大值
np.amin()
np.amax()

a=np.array([[3,7,5,4],[8,4,3,2],[2,4,9,2]])
np.amin(a,1)
np.amin(a)#平铺后所有的最小的
np.amin(a,0)

#计算最大最小值的差
np.ptp(a)
np.ptp(a,0)
np.ptp(a,1)


#百分位数：表示小于这个值的观察值的百分比
np.percentile(a,q,axis)
#q: 要计算的百分位数，在 0 ~ 100 之间,如果q=50，那么计算的就是中位数


#中位数,均数
np.median(a,axis)

np.mean(a)

#加权平均：
# 加权平均值 = (1*4+2*3+3*2+4*1)/(4+3+2+1)
np.average(a,weights=[4,2,3],returned=True,axis=0)


#标准差/方差
np.std(a)
np.var(a)


# numpy排序、条件筛选函数
np.sort()
np.argsort()
np.lexsort()

np.lexsort()
#用于对多个序列进行排序。把它想象成对电子表格进行排序，每一列代表一个序列，排序时优先照顾靠后的列。
#这里举一个应用场景：小升初考试，重点班录取学生按照总成绩录取。在总成绩相同时，数学成绩高的优先录取，在总成绩和数学成绩都相同时，按照英语成绩录取…… 这里，总成绩排在电子表格的最后一列，数学成绩在倒数第二列，英语成绩在倒数第三列。
np.partition()
#a=np.array([2,1,3,9,5,8,4])
#np.partition(a,2)
#Out[24]: array([1, 2, 3, 9, 5, 8, 4])
np.argmax()
np.argmin()

np.nonzero()
np.where()
np.extract()


# 线性代数
np.dot()#两个数组点积
np.vdot()#两个向量点积
np.inner()#内积：对应位置相乘
a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])

np.dot(a,b)#[[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]
np.vdot(a,b)#1*11 + 2*12 + 3*13 + 4*14 = 130
np.inner(np.array([1,2,3]),np.array([0,1,0]))# 1*0+2*1+3*0
np.matmul(a,b)
#若a是一维，则根据b扩展维度，相乘后取消1的维度.
#若a大于2维，则广播原则
a的shape:2,2,4
b的shape:2,4,2
c的shape:1,4,2
np.matmul(a,b)的shape2，2，2
np.matmul(a,b)的shape2，2，2




