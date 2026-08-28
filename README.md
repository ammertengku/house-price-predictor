# Malaysian House Price Predictor

This beginner-friendly Jupyter Notebook builds a complete regression workflow from a fictional Malaysian-style housing dataset. It generates the data inside the notebook, trains a random forest, evaluates it on unseen examples, draws two charts, and provides both a plain Python prediction function and an interactive form.

The dataset and results are simulated learning examples. They are not professional property valuations and should not guide a real purchase, sale, loan, or investment decision.

## Windows setup

1. Put `House_Price_Predictor.ipynb`, `requirements.txt`, and this README in a folder you can find.
2. Open Command Prompt in that folder. In File Explorer, click the folder's address bar, type `cmd`, and press Enter.
3. Run: `py -m pip install -r requirements.txt`
4. Run: `jupyter notebook`
5. Open `House_Price_Predictor.ipynb` in the browser window that appears.
6. Select **Kernel > Restart & Run All** and wait for every cell to finish.

On macOS or Linux, use `python3 -m pip install -r requirements.txt` and then `jupyter notebook` from a terminal in the project folder.

## Using the predictor

1. Wait until all cells finish.
2. Scroll to **Interactive Predictor**.
3. Choose the house details.
4. Click **Predict Price**.

Important: The data are fictional and the result is only an educational estimate.

If the interactive controls do not appear, use the `predict_house_price(...)` example in the section immediately above them. Change its values, then run that cell again.

## Troubleshooting

- **A package is missing:** In a new notebook cell, run `%pip install numpy pandas scikit-learn matplotlib ipywidgets`, then select **Kernel > Restart Kernel** and run all cells again.
- **Jupyter is not recognized:** Run `py -m notebook` on Windows instead of `jupyter notebook`.
- **The controls are blank or disabled:** Restart the kernel, run all cells in order, and use the plain function while checking the widget installation.
- **A red validation message appears:** Use one of the listed location and property types and keep every number within the range shown by the controls. Bedrooms, bathrooms, property age, and parking spaces must be whole numbers.
- **The notebook looks stuck:** Model training can take a short while. A `[*]` beside a cell means it is still working.

## Beginner exercises

After the notebook runs successfully, try changing one thing at a time:

1. Predict the same house in each location type. Record how the simulated estimates differ and explain why the generated formula causes that pattern.
2. Keep every input fixed, then compare a new home with a 40-year-old home. Relate the result to the age feature in the educational data.
3. Change `random_state=42` to another integer in the dataset call, rerun all cells, and compare R². Explain why the exact score changes even though the overall workflow stays the same.
4. Add a new fictional location category and a corresponding bonus. Update both the data generator and the validation choices, then rerun the notebook.

