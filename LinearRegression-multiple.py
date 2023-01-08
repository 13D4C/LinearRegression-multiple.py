//libray ที่เราจะใช้มี 3 อย่างก็คือ 
1.Pandas มาใช้อ่านข้อมูล 
2.Matplotlib จะใช้ในการสร้างกราฟ 
3.ผมโหลด sklearn เพื่อโหลดโมเดล มาใช้ ซึ่งมี2ตัว LinearRegression เพื่อใช้คาดคะเนโอกาสในอนาคต ส่วน metrics ใช้ตรวจสอบโอกาสคคาดเคลื่อนของการคาดคะเน
//
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

//อ่านไฟล์ CSV//
df = pd.read_csv('/kaggle/input/advertising-linear-regression/advertising.csv')

//ใช้หาค่าสหสัมพันธ์//
df.corr()

//กำหนดตัวแปร//
X = df[['TV','Radio','Newspaper']]
y = df['Sales']

//นำตัวแปรต้น-ตามใส่่ข้อมูลเข้าไปเพื่อเป็นแบบจำลอง(Train ข้อมูล)//
lrm = LinearRegression()
lrm.fit(X,y)

//พยากรณ์ข้อมูล//
predictions = lrm.predict(X)
predictions

//หาความคลาดเคลื่อนของการคาดคะเน//
metrics.mean_absolute_error(y,predictions)

//ต่อไปพยากรณ์ตัวแปรต้นแล้วจะเกิดตัวแปรตามเท่าไหร่แบบหลายตัว//
lrm.predict([[10,30,40]])
