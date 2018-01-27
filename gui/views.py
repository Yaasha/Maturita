from django.shortcuts import render, redirect

import os
import sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import backend


def index(request):
    return redirect('download.html')


def download(request):
    data = backend.handle_request(request)
    return render(request, 'gui/download.html', data)


def read(request):
    data = backend.handle_request(request)
    return render(request, 'gui/read.html', data)

# Create your views here.
