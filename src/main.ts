import * as tf from '@tensorflow/tfjs';

// Create a simple model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training. (y = 2x - 1)
const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const ys = tf.tensor2d([1, 0, -1, -2.5, -3, -3], [6, 1]);

const ueTypes = [
  "APPROVE",
  "CANCEL",
  "CHANGEPASSWORD",
  "COPY",
  "CREATE",
  "DELETE",
  "DROPSHIP",
  "EDIT",
  "EDITFORECAST",
  "EMAIL",
  "MARKCOMPLETE",
  "ORDERITEMS",
  "PACK",
  "PAYBILLS",
  "PRINT",
  "QUICKVIEW",
  "REASSIGN",
  "REJECT",
  "SHIP",
  "SPECIALORDER",
  "TRANSFORM",
  "VIEW",
  "XEDIT"
];

tf.oneHot(tf.tensor1d(["APPROVE", "EMAIL", "EMAIL", "VIEW", "EMAIL"], 'int32'), ueTypes.length).print();

// Train the model using the data.
model.fit(xs, ys, {epochs: 250}).then(() => {
  console.log(model.predict(tf.tensor2d([20], [1, 1])).toString());
});