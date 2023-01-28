import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as papaparse from 'papaparse';
import * as path from "path";

async function readCSV(): Promise<[string[], string[]]> {
  console.log('READING FILE NOW');
  const data = fs.readFileSync(path.join(__dirname, './malaysian-names.csv'), 'utf8');
  const results = papaparse.parse(data, { header: true });
  // @ts-ignore
  const names = results.data.map(row => row.name);
  // @ts-ignore
  const labels = results.data.map(row => row.country);
  return [names, labels];
}

async function main() {
  const [names, labels] = await readCSV();

  console.log("NAMES", names);
  console.log("LABELS", labels);

  const vocabularySize = names.length;
  const embeddingSize = 128;
  const numLabels = 1;

  const model = tf.sequential();
  model.add(tf.layers.embedding({
    inputDim: vocabularySize,
    outputDim: embeddingSize,
    inputLength: 1,
  }));
  model.add(tf.layers.dense({ units: numLabels, activation: 'sigmoid' }));
  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });


  const xs = tf.tensor2d(names, [names.length, 1], "string");
  const ys = tf.tensor1d(labels);
  const history = await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 32,
  });
  console.log(history.history.loss[0]);
}

main().then(() => console.log('Model all trained!'));
