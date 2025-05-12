# First-Draft

---
End of Phase 3 Specials
---

---
notebook Requirements:
---

1. Overview
2. Business and Data Understanding
3. Explain your stakeholder audience and dataset choice here
4. Modeling
5. Evaluation
6. Conclusion

- For this project, you must be modeling a classification problem.

- `Rem`: Categorical data may be represented in the data as numbers, e.g. 0 and 1, but they are not truly numeric values. If you're unsure, ask yourself "is a target value of 1 one more than a target value of 0"; if it is one more, that is a regression target, if not, that is a classification target.

- For this project we shall still use the primarily descriptive and inferential techniques, but make sure you are also using a predictive approach. This means that we shall be trying to understand the distributions of variables and the relationship between them. However, for this project we can still use these techniques, but make sure you are also using a predictive approach.

- A predictive finding might include:

> How well your model is able to predict the target
> What features are most important to your model

- A predictive recommendation might include:

> The contexts/situations where the predictions made by your model would and would not be useful for your stakeholder and business problem
> Suggestions for how the business might modify certain input variables to achieve certain target results


---
Iterative Approach to Modeling
---

- You should demonstrate an iterative approach to modeling. This means that you must build multiple models. Begin with a basic model, evaluate it, and then provide justification for and proceed to a new model. 

- `After you finish refining your models, you should provide 1-3 paragraphs in the notebook discussing your final model.`

- With the additional techniques you have learned in Phase 3, be sure to explore:

> Model features and preprocessing approaches
> Different kinds of models (logistic regression, decision trees, etc.)
> Different model hyperparameters
> At minimum you must build two models:
	-- A simple, interpretable baseline model (logistic regression or single decision tree)
	-- A version of the simple model with tuned hyperparameters


---
Classification Metrics
---

- You must choose appropriate classification metrics and use them to evaluate your models. Choosing the right classification metrics is a key data science skill, and should be informed by data exploration and the business problem itself. You must then use this metric to evaluate your model performance using both training and testing data.


---
Non-Technical Presentation
---

- Recall that the non-technical presentation is a slide deck presenting your analysis to business stakeholders, and should be presented live as well as submitted in PDF form on Canvas.

- We recommend that you follow this structure, although the slide titles should be specific to your project:

> Beginning
	-- Overview
	--- Business and Data Understanding
> Middle
	-- Modeling
	-- Evaluation
> End
	-- Recommendations
	-- Next Steps
	-- Thank you

- The discussion of classification modeling is geared towards a non-technical audience! Assume that their prior knowledge of machine learning is minimal. 

- You don't need to explain the details of your model implementations, but you should explain why classification is useful for the problem context. Make sure you translate any metrics or feature importances into their plain language implications.


---
Jupyter Notebook
---

- You will submit the notebook in PDF format on Canvas as well as in .ipynb format in your GitHub repository.

- The graded elements for the Jupyter Notebook are:

> Business Understanding
> Data Understanding
> Data Preparation
> Modeling
> Evaluation
> Code Quality


---
GitHub Repository
---

- The README.md file should be the bridge between your non technical presentation and the Jupyter Notebook. `It should not contain the code used to develop your analysis`, but should provide a more in-depth explanation of your methodology and analysis than what is described in your presentation slide

- In the README.md file should contain:

> Overview
> Business and Data Understanding
> Explain your stakeholder audience and dataset choice here
> Modeling
> Evaluation
> Conclusion