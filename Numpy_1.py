#!/usr/bin/env python
# coding: utf-8

# # <h1><center>Numpy</center></h1>

# In[1]:


import numpy as np


# # <h2><center>1-D array</center></h2>
# 

# In[2]:


l1=[1,2,3,5,6,8,9]


# In[3]:


arr1=np.array(l1)


# In[4]:


arr1


# In[5]:


l2=[2,3,4,"hello"]


# In[6]:


arr2=np.array(l2)


# 
# # <h2><center>2-D array<center><h2>

# In[7]:


l3=[[1,2,3],[5,6,7],[4,6,9]]


# In[8]:


arr3=np.array(l3)


# In[9]:


arr3


# In[10]:


l4=[["Mathematics","English","Hindi","Sanskrit"],[90,89,77,56],[66,73,34,54],[56,44,64,76]]


# In[11]:


arr4=np.array(l4)


# In[12]:


arr4


# 
# # <h2><center>3-D array<center><h2>

# In[13]:


l5=[[[1,2],[3,6,]],[[3,7],[6,9]]]


# In[14]:


arr5=np.array(l5)


# In[15]:


arr5


# In[16]:


l6=[[[1,2,6],[3,6,11],[4,8,31]],[[3,7,5],[6,9,23],[3,66,3]]]


# In[17]:


arr6=np.array(l6)


# In[18]:


arr6


# # <h2><center>Empty array<center><h2>

# In[19]:


arr7=np.empty((4,5))


# In[20]:


arr7


# In[21]:


arr8=np.empty((3,3))


# In[22]:


arr8


# In[23]:


"---------------------------------------------------------------------------------------------"


# # <h2><center>Zero array<center><h2>

# In[24]:


arr9=np.zeros((3,3))


# In[25]:


arr9 


# In[26]:


arr10=np.zeros((3,3),dtype=int)


# In[27]:


arr10


# In[28]:


arr11=np.ones((3,3))


# In[29]:


arr11


# In[30]:


arr12=np.ones((3,3),dtype=int)


# In[31]:


arr12


# # ----------------------------------------------------------------------------------------

# In[32]:


arr13=np.arange(2,8)


# In[33]:


arr13


# In[34]:


arr14=np.arange(3,23,4)


# In[35]:


arr14


# In[36]:


arr15=np.linspace(0,20,5)


# In[37]:


arr15


# In[38]:


arr16=np.linspace(0,10,9)


# In[39]:


arr16


# In[40]:


arr17=np.logspace(2,9,10)


# In[41]:


arr17


# In[42]:


arr18=np.full((3,3),12)


# In[43]:


arr18


# In[44]:


arr19=np.random.random((3,3))


# In[45]:


arr19


# # <h2><center>Numpy attributes<center><h2>

# In[46]:


print(arr1.ndim)


# In[47]:


print(arr2.ndim)


# In[48]:


for i in range(1,20):
    print(f"The dimension of array{i} is:")
    print(eval(f"arr{i}.ndim"))


# In[49]:


for i in range(1,20):
    print(f"The shape of array{i} is:")
    print(eval(f"arr{i}.shape"))


# In[50]:


for i in range(1,20):
    print(f"The size of array{i} is:")
    print(eval(f"arr{i}.size"))


# In[51]:


for i in range(1,20):
    print(f"The datatype of each element of array{i} is:")
    print(eval(f"arr{i}.dtype"))


# In[52]:


for i in range(1,20):
    print(f"The itemsize in bytes of each element of array{i} is:")
    print(eval(f"arr{i}.itemsize"))


# In[53]:


for i in range(1,20):
    print(f"The itemsize in bits of each element of array{i} is:")
    x=eval(f"arr{i}.itemsize")
    print(x/8)


# # <h2><center>Sorting<center><h2>

# In[54]:


l7=[[78,67,56],[76,75,47],[84,59,60],[67,72,54]]
marks=np.array(l7)
marks


# In[55]:


marks.sort(axis=1) 
#column wise sorting
# Changes will occur in the original array


# In[56]:


marks


# In[57]:


l7=[[78,67,56],[76,75,47],[84,59,60],[67,72,54]]


# In[58]:


marks=np.array(l7)


# In[59]:


print(marks)


# In[60]:


marks.sort(axis=0)


# In[61]:


marks


# # <h1><center>Indexing and Slicing<center><h1>

# In[62]:


arr1[0]


# In[63]:


arr1[4]


# In[64]:


marks[2,2]


# In[65]:


marks[0:3,0:2]


# In[66]:


arr1[::-1]


# In[67]:


arr2[::-1]


# In[68]:


arr15[::-1]


# In[69]:


arr8[:,0:2]


# # <h2><center>Operation on Arrays<center><h2>

# In[70]:


marks


# In[71]:


marks1=np.array([[10,20,30],[34,33,46],[34,56,76],[55,45,87]])


# In[72]:


marks1


# In[73]:


marks2=np.array([[10,20,30,88],[34,33,46,15],[34,56,76,66],[55,45,87,89]])


# In[74]:


marks3=np.array([[11,23,30,88],[37,33,46,34],[34,65,76,66],[55,35,87,89]])


# In[75]:


marks,marks1,marks2,marks3


# In[76]:


marks+marks1


# In[77]:


marks-marks1


# In[78]:


marks*marks1


# In[151]:


marks/marks1


# In[154]:


np.where(marks<50,True,False) #yield True otherwise yield False


# In[156]:


marks1


# In[ ]:





# 
# # <h2><center>Matrix Multiplication<center><h2>

# In[80]:


marks2@marks3


# # <h3><center>The Linear Algebra module of NumPy offers various methods to apply linear algebra on any numpy array.<center><h3>

# In[81]:


marks0=marks


# # <h3><center>Rank of a matrix<center><h3>

# In[82]:


for i in range(4):
    rank=eval(f"np.linalg.matrix_rank(marks{i})")
    mat=eval(f"marks{i}")
    print(f"The rank of the matrix \n{mat} is {rank}")


# # <h3><center>Determinant of a matrix<center><h3>

# In[83]:


for i in range(4):
    mat=eval(f"marks{i}")
    if mat.shape[0]==mat.shape[1]:
        det=eval(f"np.linalg.det(marks{i})")
        print(f"The determinant of the matrix \n{mat} is {det}")


# # <h3><center>Trace of a matrix<center><h3>

# In[84]:


for i in range(4):
    mat=eval(f"marks{i}")
    if mat.shape[0]==mat.shape[1]:
        tra=eval(f"np.trace(marks{i})")
        print(f"The trace of the matrix \n{mat} is {tra}")


# # <h3><center>Inverse of a matrix<center><h3>

# In[85]:


for i in range(4):
    mat=eval(f"marks{i}")
    if mat.shape[0]==mat.shape[1]:
        inverse=eval(f"np.linalg.inv(marks{i})")
        print(f"The inverse of the matrix \n{mat} is \n{inverse}")


# In[86]:


for i in range(4):
    mat=eval(f"marks{i}")
    if mat.shape[0]==mat.shape[1]:
        inverse=eval(f"np.linalg.matrix_power(marks{i},3)")
        print(f"The matrix \n{mat} raised to power 3 is  is \n{inverse}")


# # <h2><center>Eigenvalues and Eigenvectors of a matrix<center><h2>

# In[87]:


arr20=np.diag((2,3,4))


# In[88]:


arr20


# In[89]:


np.linalg.inv(arr20)


# In[90]:


np.linalg.eig(arr20) # eigenvalues and right eigenvectors


# In[91]:


np.linalg.eig(marks2) # eigenvalues and right eigenvectors


# # <h2><center>Soultion of the system of linear equations<center><h2>

# In[92]:


arr21=np.array([1,2,3,4] )
marks2


# In[93]:


np.linalg.solve(marks2,arr21)


# In[94]:


for i in range(4):
    mat=eval(f"marks{i}")
    arr21t=np.transpose(arr21)
    if mat.shape[0]==mat.shape[1]:
        x=eval(f"np.linalg.solve(marks{i},arr21)")
        print(f"The solution of the system of linear equations\n {mat}X= {arr21t} is \n{x}")


# # <h2><center>Sorting and Transpose<center><h2>

# In[95]:


marks1,np.transpose(marks1)


# In[96]:


marks2,np.transpose(marks2)


# # <h2><center>Concatenation<h2><center>

# In[97]:


marks2,marks3


# In[98]:


np.concatenate((marks2,marks3),axis=1)


# In[99]:


np.concatenate((marks2,marks3),axis=0)


# # <h2><center>Reshaping of Arrays<center><h2>

# In[100]:


arr1


# In[101]:


arr1.size


# In[102]:


arr2.size


# In[103]:


arr2


# In[104]:


arr2.reshape(2,2)


# # <h2><center> Splitting Arrays<h2><center>

# In[105]:


marks2,marks3


# In[106]:


first,second,third=np.split(marks2,[1,3],axis=0)


# In[107]:


first,second,third


# In[108]:


first,second,third=np.split(marks2,[1,3],axis=1)


# In[109]:


first,second,third


# In[110]:


first,second=np.split(marks2,2,axis=1)


# In[111]:


first,second 


# In[112]:


first,second=np.split(marks2,2,axis=0)


# In[113]:


first,second


# # <h2><center>Statistical Operations on Arrays<h2><center >

# In[114]:


marks2,marks3


# In[115]:


marks2.sum(axis=1) #axis=1 means the sum of each row


# In[116]:


marks2.sum(axis=0) #axis=0 means the sum of each column


# In[117]:


marks3.max(axis=1)  #axis=1 means the sum of each row


# In[118]:


marks3.max(axis=0) #axis=0 means the sum of each column


# In[119]:


marks2.mean(axis=1) #axis=1 means the mean of each row


# In[120]:


marks2.mean(axis=0) #axis=0 means the mean of each column


# In[121]:


marks3.std(axis=1) #axis=1 means the standard drviation of each row


# In[122]:


marks3.std(axis=0) #axis=0 means the standard drviation of each column


# # <h2><center>Random Generation and random  module<center><h2>

# In[123]:


np.random.randint(1,101,200)  #it will generate 200 random inetegers from 1(inclusive) to 200(exclusive)


# In[124]:


np.random.randint(1,20,100) #it will generate 200 random inetegers from 1(inclusive) to 20(exclusive)


# In[125]:


np.random.rand(2,3) #It will generate random number between 0 and 1 in 2*3 matrix


# In[126]:


np.random.rand(3,3,3)


# In[127]:


np.random.randint(100,size=(4,4)) #from 0 to 99


# In[128]:


np.random.choice([3, 5, 7, 9], size=(3, 5)) #It will chose numbers from the given array and generate a 3 *5 matrix


# In[129]:


np.random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))


# In[130]:


np.random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))


# In[131]:


np.random.shuffle(arr1) # make changes permanently


# In[132]:


arr1


# In[133]:


print(arr2)
print(np.random.permutation(arr2)) # will not change permanently
print(arr2)


# In[134]:


np.random.normal(loc=0,scale=1,size=(10)) # will generate from N(0,1)


# In[135]:


import seaborn as sns


# In[136]:


sns.distplot(np.random.normal(loc=0,scale=1,size=(1000)),hist=False)


# In[137]:


bin1=np.random.binomial(n=10,p=0.9,size=1000)
sns.distplot(bin1,hist=True)


# In[138]:


sns.distplot(np.random.normal(loc=50, scale=5, size=1000), hist=False, label="normal")
sns.distplot(np.random.binomial(n=100, p=0.5, size=1000), hist=False, label="binomial")


# In[139]:


arr22=np.loadtxt("Studentdata.txt",delimiter=",",skiprows=1)


# In[140]:


arr22


# In[141]:


arr23=np.loadtxt("Studentdata.txt",delimiter=",",skiprows=1,dtype=int)


# In[142]:


arr23


# In[143]:



arr24=np.genfromtxt("Studentdata1.txt",delimiter=",",skip_header=1)


# In[144]:


arr24


# In[145]:


arr25=np.genfromtxt("Studentdata1.txt",delimiter=",",skip_header=1,filling_values=55,dtype=int)


# In[146]:


arr25


# In[147]:


np.savetxt("Studentdata2.txt",arr25,delimiter=",",fmt="%i")


# In[148]:


ls


# In[149]:


arr26=np.loadtxt("studentdata2.txt",delimiter=",",dtype=int)


# In[150]:


arr26

