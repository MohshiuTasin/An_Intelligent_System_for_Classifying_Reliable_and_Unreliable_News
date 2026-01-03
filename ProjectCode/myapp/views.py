# detector/views.py

from django.shortcuts import render
from .ml_model import model, vectorizer

def index(request):
    if request.method == 'POST':
        news = request.POST['news']
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)
        val=""
        if prediction==[1]:
            val = "real"
        else:
            val = "fake" 
        return render(request, 'result.html', {'prediction': val})
    return render(request, 'index.html')
