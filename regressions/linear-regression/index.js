// require('@tensorflow/tfjs-node-gpu');
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
    shuffle: true,
    splitTest: 50,
    dataColumns: ["horsepower", "weight", "displacement"],
    labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 10,
    batchSize: 10,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
    x: regression.mseHistory.reverse(),
    xLabel: "Iteraction #",
    yLabel: "Mean Squared Error",
});

console.log("R2:", r2);
console.log("Weights:", regression.weights.arraySync());

regression.predict([
    [120, 2, 380],
    [135, 2.18, 420],
]).print();
