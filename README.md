# Regression

This repository contains Python code to perform 1-D regression with:
-   [Gaussian Process Regression](https://en.wikipedia.org/wiki/Kriging),
-   [Relevance Vector Machine](https://en.wikipedia.org/wiki/Relevance_vector_machine).

## Requirements

-   Install the latest version of [Python 3.X](https://www.python.org/downloads/).
-   Install the required packages:

```bash
pip install -r requirements.txt
pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
```

## Usage

```bash
python main.py
```

## Results

The ground truth is the sinc function.

### Influence of the number of training samples

The variable `noise_level` is set to `0.1`.
The variable `training_data_range` is set to a large value (`15`).

The results are shown with increasing number of training samples.

![10 samples](https://github.com/woctezuma/regression/wiki/img/NM0Gerr.png)
![50 samples](https://github.com/woctezuma/regression/wiki/img/9JbUXcK.png)
![100 samples](https://github.com/woctezuma/regression/wiki/img/pEmUJyn.png)
![200 samples](https://github.com/woctezuma/regression/wiki/img/r5yjsGD.png)
![500 samples](https://github.com/woctezuma/regression/wiki/img/J4krNnB.png)

### Influence of the noise level

The variable `num_samples` is set to `100`.
The variable `training_data_range` is set to a large value (`15`).

The results are shown with increasing noise level.

![noise 0.0](https://github.com/woctezuma/regression/wiki/img/lq63j83.png)
![noise 0.1](https://github.com/woctezuma/regression/wiki/img/aw7O2KS.png)
![noise 0.5](https://github.com/woctezuma/regression/wiki/img/d9dknjW.png)
![noise 1.0](https://github.com/woctezuma/regression/wiki/img/dFyDuDE.png)

### Influence of range of training data

The variable `num_samples` is set to `100`.
The variable `noise_level` is set to `0.1`.

The results are shown with increasing range of training data

![range 1](https://github.com/woctezuma/regression/wiki/img/7OyCgMI.png)
![range 2](https://github.com/woctezuma/regression/wiki/img/az6BP26.png)
![range 5](https://github.com/woctezuma/regression/wiki/img/fLn9jmF.png)
![range 10](https://github.com/woctezuma/regression/wiki/img/7tCFWFZ.png)
![range 15](https://github.com/woctezuma/regression/wiki/img/ugN0nQi.png)

## References

-   Python module [scikit-learn](https://github.com/scikit-learn/scikit-learn)
-   Documentation: [Gaussian Process](https://scikit-learn.org/stable/modules/gaussian_process.html) with scikit-learn
-   Python module [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm)
-   Python module [sklearn-rvm](https://github.com/Mind-the-Pineapple/sklearn-rvm)
-   Slides about [Relevance Vector Regression](http://lasa.epfl.ch/teaching/lectures/ML_MSc_Advanced/Slides/Lec_IX_NonlinearRegression_Part_I.pdf)
-   Slides about [Gaussian Process Regression](http://lasa.epfl.ch/teaching/lectures/ML_MSc_Advanced/Slides/Lec_IX_NonlinearRegression_Part_II.pdf)
