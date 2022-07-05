from sklearn.svm import SVC
import numpy as np

"""
SVC参数解释
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
（6）probablity: 可能性估计是否使用(true or false)；是否启用概率估计。 这必须在调用fit()之前启用，并且会fit()方法速度变慢
（7）shrinking：是否进行启发式；
（8）tol（default = 1e - 3）: svm停止训练的误差精度;
（9）cache_size: 制定训练所需要的内存（以MB为单位）；默认为200MB。
（10）class_weight: 给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C.如果给定参数‘balanced’，则使用y的值自动调整与输入数据中的类频率成反比的权重。缺省的话自适应；
（11）verbose: 是否启用详细输出。 此设置利用libsvm中的每个进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。一般情况都设为False，不用管它。
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
 ps：7,8,9一般不考虑。
 主要调节的参数有：C、kernel、degree、gamma、coef0。
 ★fit()方法：用于训练SVM，具体参数已经在定义SVC对象的时候给出了，这时候只需要给出数据集X和X对应的标签y即可。
 ★predict()方法：基于以上的训练，对预测样本T进行类别预测，因此只需要接收一个测试集T，该函数返回一个数组表示个测试样本的类别。
"""
X = [[2, 3, 7, 8],
     [4, 6, 8, 9],
     [7, 6, 4, 2]
     ]

y = [1, 0, 1]

test_dataset = [[2, 5, 6, 2],
                [4, 9, 0, 3],
                [7, 2, 7, 2]
                ]

test_labels = [
    [1],
    [1],
    [0]
]

clf = SVC(kernel='rbf')  # 调参
clf.fit(X, y)  # 训练
print(clf.fit(X, y))  # 输出参数设置
p = 0  # 正确分类的个数
for i in range(len(test_dataset)):  # 循环检测测试数据分类成功的个数

    print(clf.predict(np.array([test_dataset[i]])))
    if clf.predict(np.array([test_dataset[i]])) == test_labels[i][0]:
        p += 1

print(len(test_labels))
print(p / len(test_labels))  # 输出测试集准确率
