const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {

    const {mean, variance} = tf.moments(features, 0);

    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

    return (
        features
            .sub(mean)
            .div(variance.pow(0.5))
            .sub(scaledPrediction)
            .pow(2)
            .sum(1)
            .pow(0.5)
            .expandDims(1)
            .concat(labels, 1)
            .unstack()
            .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
            .slice(0, k)
            .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
    );
}

let { features, labels, testFeatures, testLabels } = loadCSV("kc_house_data.csv", {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
    labelColumns: ["price"],
});

// console.log('features', features);
// console.log('labels', labels);
// console.log('testFeatures', testFeatures);
// console.log('testLabels', testLabels);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    console.log("Error", err * 100);
});
