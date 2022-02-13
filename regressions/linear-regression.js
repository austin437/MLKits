const tf = require("@tensorflow/tfjs");

class LinearRegression {
    constructor(features, labels, options) {
        this.features = features;
        this.labels = labels;

        //modify to use this.options = {learningRate: 0.1, someDefaultVal: 3.4, ...options};
        this.options = Object.assign({ learningRate: 0.1 }, options);
    }
}

module.exports(LinearRegression);
