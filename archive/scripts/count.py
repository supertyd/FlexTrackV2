from collections import Counter

# 读取文件内容
with open('count.txt', 'r') as file:
    lines = file.readlines()

# 提取所有数字
numbers = []
frame_number = 0
for line in lines:
    # 假设每行都是一个列表，列表中包含数字
    numbers.extend([int(num) for num in line.strip().strip('[]').split(',')])
    frame_number = frame_number +1

# 计算每个数字的出现频率
number_counts = Counter(numbers)


average_dim = 0
dim_all = 0
counts_all =0
# 输出结果
for number, count in number_counts.items():
    if number == 0:
        dim_all = 4*count + dim_all
    if number == 1:
        dim_all = 8*count + dim_all
    if number == 2:
        dim_all = 16*count + dim_all
    if number == 3:
        dim_all = 32*count + dim_all
    if number == 4:
        dim_all = 64*count + dim_all
    if number == 5:
        dim_all = 128*count + dim_all
    if number == 6:
        dim_all = 256*count + dim_all
    if number == 7:
        dim_all = 512*count + dim_all




    print(f"Number {number} appears {count} times")
print(dim_all/frame_number)


# Number 5 appears 67 times
# Number 7 appears 28 times
# Number 2 appears 22 times
# Number 3 appears 15 times                                            0.9
# Number 1 appears 14 times
# Number 4 appears 4 times 
# 321.49333333333334       0.3 missing ratio



# 00404_UAV_outdoor6 , fps:8.113625530209143
# Number 1 appears 35 times
# Number 7 appears 17 times
# Number 5 appears 77 times
# Number 0 appears 1 times
# Number 4 appears 4 times                                        0.7
# Number 3 appears 10 times
# Number 6 appears 6 times
# Number 2 appears 26 times
# 242.86363636363637




# Number 5 appears 85 times
# Number 7 appears 15 times
# Number 2 appears 30 times
# Number 3 appears 19 times      
# Number 1 appears 22 times                                                 0.5
# Number 4 appears 3 times
# Number 6 appears 14 times
# Number 0 appears 2 times
# 248.50526315789475     00370_UAV_outdoor6   0.3 missing ratio




# Number 5 appears 73 times
# Number 7 appears 15 times
# Number 1 appears 31 times
# Number 2 appears 18 times
# Number 3 appears 9 times                                               0.3
# Number 6 appears 2 times
# Number 4 appears 6 times
# 243.42857142857142

# 00398_UAV_outdoor6——————————————   0.6
# Number 5 appears 124 times
# Number 7 appears 23 times
# Number 3 appears 61 times
# Number 6 appears 1 times                                                  0.1
# Number 1 appears 32 times
# Number 2 appears 12 times
# Number 4 appears 3 times
# 238.25










# 00406_UAV_outdoor6——————————————
# Number 3 appears 34 times
# Number 2 appears 29 times
# Number 5 appears 77 times
# Number 7 appears 12 times
# Number 4 appears 1 times
# Number 6 appears 12 times
# Number 1 appears 11 times
# 236.0909090909091