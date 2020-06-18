from flask  import*
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
car_df=pd.read_csv('cleanedcsv.csv')

x_data=car_df[car_df.columns[0:6]]
y_data=car_df[car_df.columns[6]]

cardata=pd.read_csv('cars.csv')

labels=list(cardata.columns)
def enc(label,in_t=False):
	lbe=LabelEncoder()
	lbe.fit(cardata[label])
	res =list(lbe.classes_)
	trans=list(lbe.transform(res))
	if(in_t==False):
		return trans
	else:
		pass
def inverse(label):
	lbe=LabelEncoder()
	lbe.fit(cardata[label])
	res =list(lbe.classes_)
	return res
e_buy,e_maint,e_door,e_persons,e_lugboot,e_safety,e_class=list(map(enc,cardata.columns))
i_buy,i_maint,i_door,i_persons,i_lugboot,i_safety,i_class_=list(map(inverse,cardata.columns))

buy=LabelEncoder()
maint=LabelEncoder()
door=LabelEncoder()
lug_boot=LabelEncoder()
persons=LabelEncoder()
safety=LabelEncoder()
class_=LabelEncoder()
buy=list(buy.fit_transform(car_df['buying']))
maint=list(maint.fit_transform(car_df['maint']))
door=list(door.fit_transform(car_df['door']))
persons=list(persons.fit_transform(car_df['persons']))
lug_boot=list(lug_boot.fit_transform(car_df['lug_boot']))
safety=list(safety.fit_transform(car_df['safety']))
class_=list(class_.fit_transform(car_df['class']))
x_data=list(zip(buy,maint,door,persons,lug_boot,safety))
y_data=class_
names=['Unacc','Acc','Good','VGood']

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2)
model=KNeighborsClassifier(n_neighbors=11)
model.fit(x_train,y_train)
model.score(x_test,y_test)
joblib.dump(model,'model')
jmodel=joblib.load('model')
#pre=jmodel.predict([[0, 2, 3, 2, 1, 0]])
#print(names[pre[0]])

app=Flask(__name__)
@app.route('/')
def home():
	return render_template('index.html')
@app.route('/',methods=['POST'])
def result():
	b=i_buy.index(request.form['buy'])
	m=i_maint.index(request.form['maint'])
	d=i_door.index(request.form['doors'])
	p=i_persons.index(request.form['persons'])
	l=i_lugboot.index(request.form['lug_boot'])
	s=i_safety.index(request.form['safety'])

	pre=list(jmodel.predict([[b,m,d,p,l,s]]))
	res=i_class_[pre[0]]
	
	return render_template('index.html',result=res)

if(__name__=='__main__'):
	app.run(debug=True,port=8080)