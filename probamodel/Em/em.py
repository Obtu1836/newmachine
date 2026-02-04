import numpy as np
import pandas as pd


class Em:
    def __init__(self, pab):

        # 初始假设 pab=[P(x|A),P(x|B)]
        self.pab = pd.Series(pab, index=['a', 'b'])

    def e_step(self):
        '''
        e_step 的 Docstring

        looklike 就是似然函数的结果 总共2行 m列 m为data[0] 

        每行分别是AX 和BX, 是每轮投骰子的似然函数值 
        然后转置 (习惯) 方便后面计算

        最后计算每轮似然函数的期望 得出隐变量 每轮分别选择A和B的概率 (m,2)
        
        '''
        looklike = self.pab.apply(self._calculate_looklike)#（2,m)
        looklike = looklike.T  #(m,2)
        looklike.columns = ['a', 'b']
        hidden_variable = looklike.div(np.sum(looklike, axis=1), axis=0)
        return hidden_variable

    def _calculate_looklike(self, p):
        '''
        _calculate_looklike 的 Docstring

        :param self: 
        :param p: P是A或者B类骰子 发生x事件的概率 
                所以1-p为发生 q事件的概率
                因为self.counts的列 是q事件发生的次数 ,x事件发生的次数
                要保证对应关系
                所以 ps=[1-p,p]

        所以现在是 既有对应的概率 也有事件发生的次数 似然函数
        (1-P)**m*(p)**n (应该转log 但是 转了以后我不好验证代码实现是否正确 所以没转)
        '''

        ps = np.array([1-p, p])
        looklike = np.power(ps, self.counts)
        assert type(looklike) == pd.DataFrame
        return looklike.cumprod(axis=1).iloc[:, -1]# 只取关于x事件的

    def fit(self, data):

        df = pd.DataFrame(data)  # 转dataframe
        # 统计出 每轮投骰子 出现事件Q和事件X的数量
        self.counts = df.apply(lambda x: x.value_counts().reindex(
            [0, 1], fill_value=0), axis=1)
        self.counts.columns = ['q', 'x']

        while True:
            hidden_var = self.e_step()
            pab = self.m_step(hidden_var)

            if np.allclose(pab, self.pab):
                break
            self.pab = pab

    def m_step(self, z):
        '''
        m_step 的 Docstring
        z=每轮投掷 分别选择A B 的概率 (m,2)

        p=[[qa],[qb]
           [xa],[xb]]
        
        '''


        p = z.apply(self.fun)
        normalp = p.div(p.sum(), axis=1)
        pab = normalp.iloc[1, :] #取x的
        return pab

    def fun(self, ser):

        '''
       ser是 选择 每轮 A|B 的概率 (m,) 与数量进行广播点乘
       最终得到qA和xa的 ｜ qB 和xb的 
        '''

        ff = (ser.values)[:, None]*self.counts.values
        return ff.sum(axis=0)


def main():
    '''
    main 的 Docstring
    示例介绍 有A,B两种骰子, 骰子的形状可分为 4面体 6面体 或者8面体
    data中的1 代表事件x 投掷出的骰子点数>=2  0代表 事件q 投掷出的骰子<2
    data中每一行 表示 随机抓一个骰子 连续投掷10次 的数据 (骰子属于A或者B,具体未知)
    data 总共6行 表示一共重复了6次实验
    估算 A B 到底是几面体

    思路 
        1 如果P(x|A)能够确定 就可以估出 是几面体 如果P(x|A)=0.5 那么 A最可能是4面体
         同理 P(x|B)一样 所以待估计参数 P(x|A) 和P(x|B)

        2 因为每次选择骰子投掷时  不知道选择的具体哪类一类 所以 隐变量Z 为选择A|B的概率

        3 待 估计参数 P(x|A) P(x|B) Z

        4 EM算法流程:
            1 假设 P(x|A)和P(x|B)的值 
            2 在E步 根据假设 推导出 z
            3 在M步 根据Z 更新除 P(x|A)和P(x|B)
            4 迭代 EM步 直到 P(x|A)和P(x|B)的不再变化 或者 变化微小

    '''
    data = np.array([
        [0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    ])

    model = Em([0.6, 0.5])
    model.fit(data)

    print(model.pab)


if __name__ == '__main__':
    main()
