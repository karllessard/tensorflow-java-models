package org.tensorflow.model.examples.mnist;

import static org.tensorflow.tools.StaticApi.shapeOf;

import java.util.Arrays;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.model.examples.mnist.data.ImageBatch;
import org.tensorflow.model.examples.mnist.data.MnistDataset;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.OneHot;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.core.Scalar;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Softmax;
import org.tensorflow.op.train.ApplyGradientDescent;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

public class SimpleMnist implements Runnable {

  public static void main(String[] args) {
    MnistDataset dataset = MnistDataset.create(VALIDATION_SIZE);
    try (Graph graph = new Graph()) {
      SimpleMnist mnist = new SimpleMnist(graph, dataset);
      mnist.run();
    }
  }

  @Override
  public void run() {
    Ops tf = Ops.create(graph);
    
    // Create placeholders and variables, which should fit batches of an unknown number of images
    Placeholder<TUint8> images = tf.placeholder(TUint8.DTYPE);
    Placeholder<TUint8> labels = tf.placeholder(TUint8.DTYPE);

    // Create weights with an initial value of 0
    Shape weightShape = shapeOf(dataset.imageSize(), dataset.numClasses());
    Variable<TFloat32> weights = tf.variable(weightShape, TFloat32.DTYPE);
    Assign<TFloat32> weightsInit = tf.assign(weights, tf.zeros(tf.vector(weightShape), TFloat32.DTYPE));

    // Create biases with an initial value of 0
    Shape biasShape = shapeOf(dataset.numClasses());
    Variable<TFloat32> biases = tf.variable(biasShape, TFloat32.DTYPE);
    Assign<TFloat32> biasesInit = tf.assign(biases, tf.zeros(tf.vector(biasShape), TFloat32.DTYPE));

    // Flatten images in a single dimension and normalize their pixels as floats.
    Operand<TFloat32> normalizedImages =
        tf.math.div(
            tf.reshape(
                tf.dtypes.cast(images, TFloat32.DTYPE),
                tf.vector(shapeOf(-1, dataset.imageSize()))
            ),
            tf.scalar(255.0f)
        );

    // Map labels to one hot vectors where only the expected predictions as a value of 1.0
    Operand<TFloat32> oneHotLabels =
        tf.oneHot(
            labels,
            tf.scalar(dataset.numClasses()),
            tf.scalar(1.0f),
            tf.scalar(0.0f)
        );

    // Predict the class of each image in the batch and compute the loss
    Softmax<TFloat32> softmax =
        tf.nn.softmax(
            tf.math.add(
                tf.linalg.matMul(normalizedImages, weights),
                biases
            )
        );

    Mean<TFloat32> crossEntropy =
        tf.math.mean(
            tf.math.neg(
                tf.reduceSum(
                    tf.math.mul(oneHotLabels, tf.math.log(softmax)),
                    tf.vector(1)
                )
            ),
            tf.vector(0)
        );

    // Back-propagate gradients to variables for training
    Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
    Scalar<TFloat32> alpha = tf.scalar(LEARNING_RATE);
    ApplyGradientDescent<TFloat32> weightGradientDescent = tf.train.applyGradientDescent(weights, alpha, gradients.dy(0));
    ApplyGradientDescent<TFloat32> biasGradientDescent = tf.train.applyGradientDescent(biases, alpha, gradients.dy(1));

    // Compute the accuracy of the model
    Operand<TInt64> predicted = tf.math.argMax(softmax, tf.scalar(1));
    Operand<TInt64> expected = tf.math.argMax(oneHotLabels, tf.scalar(1));
    Operand<TFloat32> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), TFloat32.DTYPE), tf.vector(0));

    // Run the graph
    try (Session session = new Session(graph)) {

      // Initialize variables
      session.runner()
          .addTarget(weightsInit)
          .addTarget(biasesInit)
          .run();

      // Train the model
      for (ImageBatch trainingBatch : dataset.trainingBatches(TRAINING_BATCH_SIZE)) {
        try (Tensor<TUint8> batchImages = TUint8.copyOf(trainingBatch.images());
            Tensor<TUint8> batchLabels = TUint8.copyOf(trainingBatch.labels())) {
            session.runner()
                .addTarget(weightGradientDescent)
                .addTarget(biasGradientDescent)
                .feed(images.asOutput(), batchImages)
                .feed(labels.asOutput(), batchLabels)
                .run();
        }
      }

      // Test the model
      ImageBatch testBatch = dataset.testBatch();
      try (Tensor<TUint8> testImages = TUint8.copyOf(testBatch.images());
           Tensor<TUint8> testLabels = TUint8.copyOf(testBatch.labels());
           Tensor<TFloat32> accuracyValue = session.runner()
              .fetch(accuracy)
              .feed(images.asOutput(), testImages)
              .feed(labels.asOutput(), testLabels)
              .run()
              .get(0)
              .expect(TFloat32.DTYPE)) {
        System.out.println("Accuracy: " + accuracyValue.floatValue());
      }
    }
  }

  private static final int VALIDATION_SIZE = 0;
  private static final int TRAINING_BATCH_SIZE = 100;
  private static final float LEARNING_RATE = 0.2f;
  
  private Graph graph;
  private MnistDataset dataset;
  
  private SimpleMnist(Graph graph, MnistDataset dataset) {
    this.graph = graph;
    this.dataset = dataset;
  }
}
