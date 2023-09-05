from threading import Lock
from flask import Flask, render_template, request, session, redirect, url_for, g
from flask_socketio import SocketIO, emit, disconnect
from recog import Recogniser
from recog import MaskRecogniser
from flask_mysqldb import MySQL
from datetime import date
import MySQLdb as MySQLAsync
import MySQLdb.cursors
import cv2
import datetime
import base64

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

#database connection
mysql = MySQL()
mysql.init_app(app)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'studentrecog'

def background_thread():
    #Intial state values
    state = 0
    found_student_id = -1
    found_confidence = -1
    socketio.emit('state_change', { 'new_state':  'face' })
    cam = cv2.VideoCapture(0)
    r = Recogniser()
    m = MaskRecogniser()
    m.set_camera(cam)
    r.set_camera(cam)
    wait_start = -1

    while True:
        socketio.sleep(0.03)
        #Face state - detecting studentID using recogniser
        if(state == 0):
            r1, r2, image = r.get_student_id()
            socketio.emit('image_data', { 'buffer':  'data:image/jpg;base64,'+image, 'student_id' : r1, 'confidence' : r2 , 'wait_timer' : '-1'})
            if(r2 < 90) and (r2 != -1 and r1 != -1):
                print("Found Student ID: " + str(r1))
                found_student_id = r1
                found_confidence = r2
                state = 1
                socketio.emit('state_change', { 'new_state':  'mask' })
        #Mask state - detecting if student is masked using maskrecogniser
        elif(state == 1):
           _, r2, image = m.get_mask()
           socketio.emit('image_data', { 'buffer': 'data:image/jpg;base64,'+image, 'student_id' : found_student_id, 'confidence' : str(r2) , 'wait_timer' : '-1'})
           if(r2 >= 0.9999):
                state = 2
                socketio.emit('state_change', { 'new_state':  'update database' })
        #Database state - making database request to insert student into attendance register 
        elif(state == 2):
            
            #Mysql in this case has to be manually connected, flask_mysqldb will not instantiate the connection
            mysql_async = MySQLAsync.connect("localhost", "root", "", "studentrecog")
            cur = mysql_async.cursor()
            cur.execute("INSERT attendance_register SET Student_ID = "+str(found_student_id)+", Lecture_ID ='1', Present = '1'")
            mysql_async.commit()
            cur.close()
            found_student_id = -1
            found_confidence = 0 
            socketio.emit('state_change', { 'new_state':  'wait' })
            state = 3
        #Waiting state - Without waiting state, system is too quick going back to face state
        elif(state == 3):
            #Use default camera image whilst we wait
            ret, image = cam.read()
            ret, image2 = cv2.imencode('.jpg', image)
            data = base64.b64encode(image2).decode("UTF-8")

            if(wait_start == -1):
                wait_start = datetime.datetime.now()
            current_wait = (datetime.datetime.now() - wait_start).total_seconds()
            if(current_wait > 8):
                socketio.emit('state_change', { 'new_state':  'face' })
                state = 0
                wait_start = -1
            else:
                socketio.emit('image_data', { 'buffer': 'data:image/jpg;base64,'+data, 'wait_timer' : round(8-current_wait, 2)})


@app.before_request
def before_request():
    g.user = None

    if 'user_id' in session:
        user = session['user_id'] #name of user needs to be taken from databse
        g.user = user

@app.route('/')
def blank():
    return redirect(url_for('login'))

#lecture render
@app.route('/lectures')
def index():
    if session.get('logged_in') == True:
        return render_template('Lectures.html')
    else:
        return redirect(url_for('login'))
        
#homepage render
@app.route('/homepage')
def homepage():
    if session.get('logged_in') == True:
        currentDate = date.today()
        corDateT = currentDate.strftime("%Y-%m-%d %H:%M:%S")
        cur = mysql.connection.cursor()
        cur.execute("SELECT Module_Name, Start_DateTime, End_DateTime FROM lectures INNER JOIN modules ON lectures.Module_ID=modules.Module_ID Where Start_DateTime >= %s ORDER BY Start_DateTime ASC",(corDateT, ))
        mods = cur.fetchall()
        return render_template('Homepage.html', name=g.user, data=mods)
    else:
        return redirect(url_for('login'))

#enroll render with database entry of student
@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if session.get('logged_in') == True:
        if request.method == "POST":
            details = request.form
            firstName = details['firstName']
            lastName = details['lastName']
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO Students(First_Name, Last_Name) VALUES (%s, %s)", (firstName, lastName))
            mysql.connection.commit()
            cur.close()
            return redirect(url_for('index'))
        return render_template('Enroll.html')
    else:
        return redirect(url_for('login'))


#login render with database connection for login information
@app.route('/Login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.pop('user_id', None)
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM Lecturers WHERE username = %s AND password = %s",(username, password,))
        account = cursor.fetchone()
        if account:
            session['logged_in'] = True
            session['user_id'] = account['username']
            session['password'] = account['password']
            return redirect(url_for('homepage'))
        else:
            return redirect(url_for('login'))
    else:
        return render_template('Login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@socketio.event
def connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected', request.sid)


if __name__ == '__main__':
    socketio.run(app, debug=True)
