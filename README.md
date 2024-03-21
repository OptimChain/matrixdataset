Step 1 : Clone the project at your device.

Step 2: Create a virtual environment using the following command:

python -m venv <name_of_the_virtual_environment>

Step 3: Install the required dependencies from the requirements.txt file using the command:

pip install -r requirements.txt

step 4: Run migrations command. This is optional For any Db(models) changes.

    python3 manage.py makemigrations
    python3 manage.py migrate

Step 5: run the project using the command:

python3 manage.py runserver
