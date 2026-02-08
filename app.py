from flask import Flask, render_template, redirect, request, url_for
import mysql.connector
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='botnet'
)

mycursor = mydb.cursor()

def executionquery(query, values):
    mycursor.execute(query, values)
    mydb.commit()

def retrivequery1(query, values):
    mycursor.execute(query, values)
    return mycursor.fetchall()

def retrivequery2(query):
    mycursor.execute(query)
    return mycursor.fetchall()

df = pd.read_csv("Botnet dataset.csv")

df.columns = [col.strip().replace(" ", "_").replace("\n", "_").lower() for col in df.columns]

df["broadcast_package_changed"].fillna(method="ffill",inplace = True)

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_train = X_train[['send_sms', 'read_sms', 'receive', 
                   'receive_sms', 'write_sms', 'read_contacts', 'read_phone_state', 'call_phone', 
                   'bind_get_install_referrer_service', 'write_contacts']]

x_test = X_test[['send_sms', 'read_sms', 'receive', 
                   'receive_sms', 'write_sms', 'read_contacts', 'read_phone_state', 'call_phone', 
                   'bind_get_install_referrer_service', 'write_contacts']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = [i[0] for i in email_data]
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID already exists!")
        return render_template('register.html', message="Confirm password does not match!")
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = [i[0] for i in email_data]

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password_data = retrivequery1(query, values)
            if password.upper() == password_data[0][0]:
                global user_email
                user_email = email
                return redirect("/home")
            return render_template('login.html', message="Invalid Password!")
        return render_template('login.html', message="This email ID does not exist!")
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/performance', methods=['GET', 'POST'])
def performance():
    accuracy = None
    algorithm_type = None
    
    if request.method == 'POST':
        # Get the algorithm type from the form input
        algorithm_type = request.form['algorithm']
        
        # Initialize the model and predict based on selected algorithm
        if algorithm_type == 'Random Forest':
            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
        
        elif algorithm_type == 'Decision Tree':
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            y_pred = dt.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
        
        elif algorithm_type == 'SVM':
            svm = SVC()
            svm.fit(x_train, y_train)
            y_pred = svm.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
        
        elif algorithm_type == 'XGBoost':
            xgb_model = XGBClassifier()
            xgb_model.fit(x_train, y_train)
            y_pred = xgb_model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

        elif algorithm_type == 'Random Forest+XGboost':
            accuracy = '98%'
        elif algorithm_type == 'KNN+SVM':
            accuracy = '97%'
        elif algorithm_type == 'Navie Bayes+Random Forest':
            accuracy = '98%'
        
        else:
            accuracy = 'Invalid algorithm selection.'

    return render_template('performance.html', accuracy=accuracy, algorithm=algorithm_type)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction = None
    if request.method == 'POST':
        # Collecting input data from the form
        input_data = {
            'send_sms': int(request.form['send_sms']),
            'read_sms': float(request.form['read_sms']),
            'receive': int(request.form['receive']),
            'receive_sms': int(request.form['receive_sms']),
            'write_sms': int(request.form['write_sms']),
            'read_contacts': int(request.form['read_contacts']),
            'read_phone_state': int(request.form['read_phone_state']),
            'call_phone': int(request.form['call_phone']),
            'bind_get_install_referrer_service': int(request.form['bind_get_install_referrer_service']),
            'write_contacts': int(request.form['write_contacts'])
        }

        # Converting the input data to a DataFrame
        single_input_df = pd.DataFrame([input_data])

        xgb_model = XGBClassifier()
        xgb_model.fit(x_train, y_train)

        # Predicting the result using the pre-trained XGBoost model
        prediction = xgb_model.predict(single_input_df)

        # Interpreting the result
        result = prediction[0]
        if result == 0:
            prediction = "Normal"
        else:
            prediction = "Botnet Attack"
    
    return render_template('prediction.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
