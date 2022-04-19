//require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../load-csv");
const MultinominalLogisticRegression = require("./multinominal-logistic-regression");
const plot = require("node-remote-plot");
const _ = require("lodash");

const { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["mpg"],
    shuffle: true,
    splitTest: 50,
    converters: {
        mpg: (value) => {
            const mpg = parseFloat(value);

            if (mpg < 15) {
                return [1, 0, 0];
            } else if (mpg < 30) {
                return [0, 1, 0];
            } else {
                return [0, 0, 1];
            }
        },
    },
});

const regression = new MultinominalLogisticRegression(features, _.flatMap(labels), {
    learningRate: 0.5,
    iterations: 15,
    batchSize: 50,
});

function validate() {
    console.assert(regression.weights.shape[0] === regression.features.shape[1]);
    console.assert(regression.weights.shape[1] === regression.labels.shape[1]);
}

validate();

regression.train();

console.log(regression.test(testFeatures, _.flatMap(testLabels)));

// Will return the location of the column which
// contains the predicted output.
regression.predict([[150, 200, 2.40]]).print();
