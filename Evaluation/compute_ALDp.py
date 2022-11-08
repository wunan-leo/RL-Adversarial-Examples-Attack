#计算ALDp的公式

##  m为对抗样本的数量

## 计算ALDpL的计算公式

import math

def compute_ALDp(m,X,Xadv,p):

    sum1=0

    sum2=0

    ALDpL2_result=0

    for i in range(m):
        for j in range(len(Xadv)):
            sum1=sum1+math.pow(Xadv[i][j]-X[i][j],p)
            sum2=sum2+math.pow(X[i][j],p)
        ALDpL2_result = ALDpL2_result+math.pow(sum1, 1 / p) / math.pow(sum2, 1 / p)


    return ALDpL2_result/m


if __name__ == "__main__":
   m=input("对抗样本数量：")
   X=input("原始样本矩阵：")
   Xadv=input("对抗样本矩阵：")
   p=input("范数距离：")
   ans=compute_ALDp(m,X,Xadv,p)
   print("ALDp的值为"+ans)


