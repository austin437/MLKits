const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

module.exports = class LinearRegression {
    constructor(features, labels, options) {
        this.features = tf.tensor(features);
        this.labels = tf.tensor(labels);

        // create a tensor of 1s and concat onto features
        this.features = tf.ones([this.features.shape[0], 1]).concat(this.features, 1);

        //modify to use this.options = {learningRate: 0.1, someDefaultVal: 3.4, ...options};
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options);

        this.weights = tf.zeros([2, 1]);
    }

    gradientDescent() {
        const currentGuesses = this.features.matMul(this.weights);
        const differences = currentGuesses.sub(this.labels);
        const slopes = this.features.transpose().matMul(differences).div(this.features.shape[0]);
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    // manual algorithm
    // gradientDescent() {
    //     const currentGuessesForMPG = this.features.map((row) => {
    //         return this.m * row[0] + this.b;
    //     });

    //     const bSlope =
    //         (_.sum(
    //             currentGuessesForMPG.map((guess, i) => {
    //                 return guess - this.labels[i][0];
    //             })
    //         ) *
    //             2) /
    //         this.features.length;

    //     const mSlope =
    //         (_.sum(
    //             currentGuessesForMPG.map((guess, i) => {
    //                 return -1 * this.features[i][0] * (this.labels[i][0] - guess);
    //             })
    //         ) *
    //             2) /
    //         this.features.length;

    //     this.m = this.m - mSlope * this.options.learningRate;
    //     this.b = this.b - bSlope * this.options.learningRate;

    //     console.log("m:", this.m, "b:", this.b);
    // }

    train() {
        for (let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent();
        }
    }
};
