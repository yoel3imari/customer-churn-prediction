# ChatGPT analysis of Cell2Cell dataset:

The **Cell2Cell churn dataset** is commonly used in machine learning. Below is a data dictionary explaining every column and how you should treat it during EDA.

---

# 1. Identification Columns

| Column       | Meaning                    | Use?   |
| ------------ | -------------------------- | ------ |
| `Unnamed: 0` | Row index from CSV         | ❌ Drop |
| `X`          | Another row/index variable | ❌ Drop |
| `customer`   | Customer ID                | ❌ Drop |

These carry no predictive information.

---

# 2. Dataset Split

| Column      | Meaning                                                               |
| ----------- | --------------------------------------------------------------------- |
| `traintest` | Indicates whether the observation belongs to the training or test set |

Usually:

* 1 = Training
* 0 = Test

Keep it only if you want to reproduce the original split.

---

# 3. Target Variables

| Column     | Meaning                                                                  |
| ---------- | ------------------------------------------------------------------------ |
| `churn`    | Target (0 = stayed, 1 = churned)                                         |
| `churndep` | Churn probability or dependent variable available only for training data |

Notice:

```
churndep
40000 non-null
```

This corresponds to the original training set.

---

# 4. Revenue & Billing

| Column     | Meaning                                     |
| ---------- | ------------------------------------------- |
| `revenue`  | Monthly revenue                             |
| `recchrge` | Monthly recurring charges                   |
| `directas` | Direct access charges                       |
| `overage`  | Overage charges                             |
| `changer`  | Revenue change compared with previous month |

Questions during EDA:

* Do churners spend less?
* Do churners have high overage?

---

# 5. Usage Statistics

"MOU" means **Minutes of Use**.

| Column     | Meaning                   |
| ---------- | ------------------------- |
| `mou`      | Total monthly minutes     |
| `changem`  | Change in monthly minutes |
| `roam`     | Roaming minutes           |
| `mourec`   | Received minutes          |
| `outcalls` | Outgoing calls            |
| `incalls`  | Incoming calls            |
| `peakvce`  | Peak-hour calls           |
| `opeakvce` | Off-peak calls            |

---

# 6. Call Quality

| Column     | Meaning                 |
| ---------- | ----------------------- |
| `dropvce`  | Dropped calls           |
| `blckvce`  | Blocked calls           |
| `unansvce` | Unanswered calls        |
| `dropblk`  | Dropped + Blocked calls |
| `callwait` | Call waiting usage      |
| `callfwdv` | Call forwarding         |
| `threeway` | Three-way calls         |
| `custcare` | Calls to customer care  |

Very important for churn prediction.

---

# 7. Subscription Information

| Column     | Meaning                 |
| ---------- | ----------------------- |
| `months`   | Months with the company |
| `uniqsubs` | Number of subscriptions |
| `actvsubs` | Active subscriptions    |

---

# 8. Phone Information

| Column    | Meaning                  |
| --------- | ------------------------ |
| `phones`  | Number of phones         |
| `models`  | Number of handset models |
| `eqpdays` | Days using current phone |

Customers with newer devices may churn less.

---

# 9. Demographics

| Column     | Meaning                           |
| ---------- | --------------------------------- |
| `age1`     | Age of primary customer           |
| `age2`     | Age of secondary household member |
| `children` | Number of children                |
| `income`   | Income category                   |
| `incmiss`  | Income missing indicator          |

---

# 10. Credit

| Column     | Meaning                      |
| ---------- | ---------------------------- |
| `credita`  | Credit class                 |
| `creditaa` | Alternative credit indicator |
| `creditcd` | Has credit card              |

---

# 11. Geographic (One-Hot Encoded)

Only one should be 1.

| Column     | Meaning       |
| ---------- | ------------- |
| `prizmrur` | Rural         |
| `prizmub`  | Urban         |
| `prizmtwn` | Town/Suburban |

---

# 12. Device Features

Binary (0/1)

| Column     | Meaning                  |
| ---------- | ------------------------ |
| `refurb`   | Refurbished phone        |
| `webcap`   | Web-capable phone        |
| `newcelly` | New cellphone user (Yes) |
| `newcelln` | New cellphone user (No)  |

---

# 13. Ownership

Binary

| Column    | Meaning              |
| --------- | -------------------- |
| `truck`   | Owns truck           |
| `rv`      | Owns RV              |
| `pcown`   | Owns computer        |
| `ownrent` | Owns home (vs rents) |
| `mcycle`  | Owns motorcycle      |

---

# 14. Occupation (One-Hot)

Only one is usually 1.

| Column    | Meaning       |
| --------- | ------------- |
| `occprof` | Professional  |
| `occcler` | Clerical      |
| `occcrft` | Crafts        |
| `occstud` | Student       |
| `occhmkr` | Homemaker     |
| `occret`  | Retired       |
| `occself` | Self-employed |

---

# 15. Marital Status

| Column     | Meaning           |
| ---------- | ----------------- |
| `marryyes` | Married           |
| `marryun`  | Unknown/Unmarried |

---

# 16. Marketing

| Column     | Meaning                 |
| ---------- | ----------------------- |
| `mailord`  | Buys through mail order |
| `mailres`  | Responds to mail        |
| `mailflag` | Receives mail offers    |
| `travel`   | Travels frequently      |

---

# 17. Customer Retention

Very important.

| Column     | Meaning                        |
| ---------- | ------------------------------ |
| `retcalls` | Number of retention calls      |
| `retaccpt` | Accepted retention offer       |
| `retcall`  | Contacted retention department |

These often have a strong relationship with churn.

---

# 18. Referral

| Column  | Meaning                  |
| ------- | ------------------------ |
| `refer` | Number of referrals made |

---

# 19. Pricing

| Column    | Meaning             |
| --------- | ------------------- |
| `setprcm` | Price plan category |
| `setprc`  | Price plan value    |

---

# Missing Values

Your dataset has missing values in these columns:

| Column     | Missing |
| ---------- | ------: |
| `churndep` |  31,047 |
| `revenue`  |     216 |
| `mou`      |     216 |
| `recchrge` |     216 |
| `directas` |     216 |
| `overage`  |     216 |
| `roam`     |     216 |
| `changem`  |     502 |
| `changer`  |     502 |
| `phones`   |       1 |
| `models`   |       1 |
| `eqpdays`  |       1 |
| `age1`     |   1,244 |
| `age2`     |   1,244 |

---

## Recommended EDA order

A structured EDA for this dataset would be:

1. Drop non-informative columns (`Unnamed: 0`, `X`, `customer`).
2. Examine the target `churn` for class balance.
3. Analyze missing values and decide on imputation or removal.
4. Summarize numerical features (`describe()`, histograms, boxplots).
5. Explore categorical and binary features (value counts, count plots).
6. Compare each feature against `churn` (boxplots for numerical, stacked/count plots for categorical).
7. Compute correlations among numerical variables and identify multicollinearity.
8. Detect outliers in continuous features such as `revenue`, `mou`, `overage`, and `months`.
9. Summarize the main insights and identify features that may require preprocessing or feature engineering before modeling.

This workflow aligns well with professional EDA practices and is a solid foundation for building a churn prediction model.
