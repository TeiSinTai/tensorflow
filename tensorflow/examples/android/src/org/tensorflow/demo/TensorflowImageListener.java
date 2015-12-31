/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.hardware.Camera.PreviewCallback;
import junit.framework.Assert;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;

import java.util.List;

/**
 * Class that takes in preview frames and converts the image to Bitmaps to process with Tensorflow.
 */
public class TensorflowImageListener implements Camera.PreviewCallback {
  private static final Logger LOGGER = new Logger();

  private static final boolean SAVE_PREVIEW_BITMAP = true;

  private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
  private static final String LABEL_FILE =
      "file:///android_asset/imagenet_comp_graph_label_strings.txt";

  private static final int NUM_CLASSES = 1001;
  private static final int INPUT_SIZE = 224;
  private static final int IMAGE_MEAN = 117;

  // TODO(andrewharp): Get orientation programatically.
  private final int screenRotation = 270;

  private final TensorflowClassifier tensorflow = new TensorflowClassifier();

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private RecognitionScoreView scoreView;

  public void initialize(final AssetManager assetManager, final RecognitionScoreView scoreView) {
    tensorflow.initializeTensorflow(
        assetManager, MODEL_FILE, LABEL_FILE, NUM_CLASSES, INPUT_SIZE, IMAGE_MEAN);
    this.scoreView = scoreView;
  }

  private void drawResizedBitmap(final Bitmap src, final Bitmap dst) {
    Assert.assertEquals(dst.getWidth(), dst.getHeight());
    final float minDim = Math.min(src.getWidth(), src.getHeight());

    final Matrix matrix = new Matrix();

    // We only want the center square out of the original rectangle.
    final float translateX = -Math.max(0, (src.getWidth() - minDim) / 2);
    final float translateY = -Math.max(0, (src.getHeight() - minDim) / 2);
    matrix.preTranslate(translateX, translateY);

    final float scaleFactor = dst.getHeight() / minDim;
    matrix.postScale(scaleFactor, scaleFactor);

    // Rotate around the center if necessary.
    if (screenRotation != 0) {
      matrix.postTranslate(-dst.getWidth() / 2.0f, -dst.getHeight() / 2.0f);
      matrix.postRotate(screenRotation);
      matrix.postTranslate(dst.getWidth() / 2.0f, dst.getHeight() / 2.0f);
    }

    final Canvas canvas = new Canvas(dst);
    canvas.drawBitmap(src, matrix, null);
  }

  @Override
  public void onPreviewFrame(byte[] bytes, Camera camera) {
	Camera.Size previewSize = camera.getParameters().getPreviewSize();
	try {
		// Initialize the storage bitmaps once when the resolution is known.
	    if (previewWidth != previewSize.width || previewHeight != previewSize.height) {
	    	previewWidth = previewSize.width;
	        previewHeight = previewSize.height;
	
			LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
	        rgbBytes = new int[previewWidth * previewHeight];
	        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
	        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);
		}

		ImageUtils.convertYUV420SPToARGB8888(bytes, rgbBytes, previewWidth, previewHeight, false);

	} catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      return;
    }

	rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
    drawResizedBitmap(rgbFrameBitmap, croppedBitmap);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    final List<Classifier.Recognition> results = tensorflow.recognizeImage(croppedBitmap);

    LOGGER.v("%d results", results.size());
    for (final Classifier.Recognition result : results) {
      LOGGER.v("Result: " + result.getTitle());
    }
    scoreView.setResults(results);

	camera.addCallbackBuffer(bytes);
  }
}
