# nlp-assignment-3

After cloning the Git Repo

**Download Data**

Recipe-ingredients dataset: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv

Ingredient-description of each ingredient - https://foodb.ca/ 

**Env setup:**
```shell
conda create -n nlp
conda activate nlp
pip install -r requirement.txt
conda install -c conda-forge gensim=4.3.1
pip install flask==2.3.2
```

**Run the script:**
```shell
python backend.py
```
Now, while the backend.py is running, double click on the webpage.html from your file folder.


main.ipynb - Data Processing, cleaning and training
model3.bin - Trained model
