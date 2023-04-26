#Hair Swapper

## Setup

The first thing to do is to clone the repository:

```sh
$ git clone https://github.com/segestic/Hair-Face-Swapper---MachineLearn.git
$ cd Hair---MachineLearn

```

Create a virtual environment to install dependencies in and activate it:

```sh
$ python -m venv venv
$ source venv/bin/activate
or on windows
$  .\venv\scripts\activate
```

Then install the dependencies:

```sh
(venv)$ pip install -r requirements.txt
```
Note the `(venv)` in front of the prompt. This indicates that this terminal
session operates in a virtual environment.

Once `pip` has finished downloading the dependencies:
open the .env.example file and fill your own detail then #rename it to .env

.env come from Python Decouple is a Python library used to separate our configuration settings from code.


Migrate apps
```sh
(venv)$ python manage.py migrate
```

Afterwards runserver by:
```sh
(venv)$ python manage.py runserver
```

And navigate to `http://127.0.0.1:8000/`.
