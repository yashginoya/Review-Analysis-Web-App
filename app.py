from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import bagofwords
import tfidf
from tfidf import vect as vc
import word2vec
import pickle

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///textClassified.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.app_context().push()

bagOfWord_pickled_model = pickle.load(open('bagOfWords.pkl', 'rb'))
tfidf_pickled_model = pickle.load(open('tfidf.pkl', 'rb'))
word2vec_pickled_model = pickle.load(open('word2vec.pkl', 'rb'))

class Todo(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    desc = db.Column(db.String(500), nullable=False)
    model_type = db.Column(db.String(30), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.title}"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        title = request.form['title']
        temp = title
        select = request.form.get('select_Model')
        if str(select) == 'Bag of words':
            print('working')
            output = bagOfWord(temp)
            todo = Todo(title=temp, desc=output, model_type='Bag of Word')
            db.session.add(todo)
            db.session.commit()
        elif str(select) == 'TFIDF':
            output = tfidf(temp)
            todo = Todo(title=temp, desc=output, model_type='TFIDF')
            db.session.add(todo)
            db.session.commit()
        elif str(select) == 'word2vec':
            output = word2vecfun(temp)
            todo = Todo(title=temp, desc=output, model_type='word2vec')
            db.session.add(todo)
            db.session.commit()

    print('hello beta')
    allTodo = Todo.query.all() 
    return render_template('index.html', allTodo=allTodo)

@app.route('/show')
def show():
    allTodo = Todo.query.all()
    print(allTodo)
    return 'All Todos'

@app.route('/update/<int:sno>', methods=['GET', 'POST'])
def update(sno):
    if request.method == 'POST':
        title = request.form['title']
        temp = title
        select = request.form.get('select_Model')
        todo = Todo.query.filter_by(sno=sno).first()
        todo.title = title
        if str(select) == 'Bag of words':
            print('update working')
            output = bagOfWord(temp)
            todo.desc = output
            todo.model_type = 'Bag of Word'
            db.session.add(todo)
            db.session.commit()
        elif str(select) == 'TFIDF':
            output = tfidf(temp)
            todo.desc = output
            todo.model_type = 'TFIDF'
            db.session.add(todo)
            db.session.commit()
        elif str(select) == 'word2vec':
            output = word2vecfun(temp)
            todo.desc = output
            todo.model_type = 'word2vec'
            db.session.add(todo)
            db.session.commit()
        return redirect('/')

    todo = Todo.query.filter_by(sno=sno).first()
    return render_template('update.html', todo=todo)

@app.route('/delete/<int:sno>')
def delete(sno):
    todo = Todo.query.filter_by(sno=sno).first()
    db.session.delete(todo)
    db.session.commit()
    return redirect('/')

def bagOfWord(title):
    temp = title
    title = bagofwords.vect.transform([title])
    result = bagOfWord_pickled_model.predict(title)
    if int(result[0]) == 1:
        output = "Positive"
    else:
        output = "Negative"
    return output

def word2vecfun(title): 
    temp = title
    # title = bagofwords.vect.transform([title][])
    # title=word2vec.document_vector(title)
    # text = bagofwords.vect.transform([title])
    # result = word2vec_pickled_model.predict(text)
    title = word2vec.remove_tags(title)
    title = title.lower()
    title = word2vec.omk(title)
    title = word2vec.fun(title)
    title= word2vec.document_vector(title)
    result = word2vec_pickled_model.predict([title])
    # result = word2vec.predict(title)
    if int(result[0]) == 1:
        output = "Positive"
    else:
        output = "Negative"
    return output

def tfidf(title):
    temp = title
    title = vc.transform([title])
    result = tfidf_pickled_model.predict(title)
    if int(result[0]) == 1:
        output = "Positive"
    else:
        output = "Negative"
    return output

if __name__ == "__main__":
    app.run(debug=True,port=8000)