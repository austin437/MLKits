// require('@tensorflow/tfjs-node-gpu');
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
});

