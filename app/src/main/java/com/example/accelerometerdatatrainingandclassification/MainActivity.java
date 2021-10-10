package com.example.accelerometerdatatrainingandclassification;

import androidx.appcompat.app.AppCompatActivity;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    Dataset dset = new DefaultDataset();
    Classifier knn = null;

    SensorManager sm = null;
    Sensor s = null;
    ArrayList<float[]> acc = new ArrayList();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sm = (SensorManager) getSystemService(SENSOR_SERVICE);
        s = sm.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);

    }

    void processFile(InputStreamReader isr, String label) {

        String str = new String();
        ArrayList<Double> vals = new ArrayList<>(100);
        ArrayList<Double> sum = new ArrayList<>(100);
        ArrayList<Double> sum2 = new ArrayList<>(100);

        try {
            BufferedReader br = new BufferedReader(isr);
            while((str = br.readLine()) != null){
                String parts[] = str.split("\\t");
                double ax = Double.parseDouble(parts[0]);
                double ay = Double.parseDouble(parts[1]);
                double az = Double.parseDouble(parts[2]);
                double a = ax*ax+ay*ay+az*az;
                vals.add(Math.sqrt(a));
            }
            br.close();
        }catch (Exception ex){
            ex.printStackTrace();
        }

        sum.add(vals.get(0));
        sum2.add(vals.get(0) * vals.get(0));
        for(int i = 1; i < vals.size(); i++){
            sum.add(sum.get(i-1) + vals.get(i));
            sum2.add(sum2.get(i-1) + vals.get(i) * vals.get(i));
        }

        for(int i = 0; i + 10 < vals.size(); i++){
            double mean = (sum.get(i+9) - sum.get(i)) / 10;
            double stdev = Math.sqrt(((sum2.get(i+9) - sum2.get(i)) / 10) - mean*mean);
            dset.add(new DenseInstance(new double[]{mean, stdev}, label));
        }

    }

    public void trainClicked(View view) {
        
        InputStreamReader f1 = new InputStreamReader(getResources().openRawResource(R.raw.standing));
        processFile(f1, "standing");
        InputStreamReader f2 = new InputStreamReader(getResources().openRawResource(R.raw.walking));
        processFile(f2, "walking");
        InputStreamReader f3 = new InputStreamReader(getResources().openRawResource(R.raw.running));
        processFile(f3, "running");

        knn = new KNearestNeighbors(5);
        knn.buildClassifier(dset);
        Log.v("MYTAG", "Classifier created.");

    }

    public void classifyClicked(View view) {
        sm.unregisterListener(this);
        sm.registerListener(this, s, 100000);
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        acc.add(sensorEvent.values);
        if(acc.size() > 10) acc.remove(0);
        if(acc.size() >= 10){
            double val = 0, sum = 0, sum2 = 0;
            for(int i = 0; i < 10; i++){
                val = Math.sqrt(acc.get(i)[0]*acc.get(i)[0]+acc.get(i)[1]*acc.get(i)[1]+acc.get(i)[2]*acc.get(i)[2]);
                sum += val;
                sum2 += (val*val);
            }
            double mean = sum / 10;
            double stdev = Math.sqrt(sum2/10 - mean*mean);
            Instance ins = new DenseInstance(new double[]{mean, stdev});
            Object result = knn.classify(ins);
            if(result != null) {
                Log.v("MYTAG", result.toString());
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}