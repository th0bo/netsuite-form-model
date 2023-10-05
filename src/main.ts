import * as tf from '@tensorflow/tfjs';

// // Generate some synthetic data for training. (y = 2x - 1)
// const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
// const ys = tf.tensor2d([1, 0, -1, -2.5, -3, -3], [6, 1]);

const oneHotEncode = <T>(elements: T[], group: T[]) => {
  return tf.oneHot(
    tf.tensor1d(
      elements.map(element => group.indexOf(element)),
      'int32'
    ), group.length
  );
}

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

const userRoles = [
  "ADMIN",
  "SALES",
  "DEV",
];

const xs2 = oneHotEncode(["APPROVE", "EMAIL", "EMAIL", "VIEW", "EMAIL", "EMAIL", "EMAIL", "EMAIL", "EMAIL"], ueTypes);
const ys2 = oneHotEncode(["ADMIN", "SALES", "SALES", "ADMIN", "SALES", "DEV", "SALES", "SALES", "SALES"], userRoles);

// Create a simple model.
const inputs = tf.input({ shape: [ueTypes.length] });
const layer = tf.layers.dense({units: 3, inputShape: [ueTypes.length]});
const outputs = layer.apply(inputs) as tf.SymbolicTensor[];
const model = tf.model({ inputs, outputs });

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Train the model using the data.
model.fit(xs2, ys2, {epochs: 250}).then(() => {
  console.log(model.predict(oneHotEncode(['EMAIL', 'APPROVE'], ueTypes)).toString());
});