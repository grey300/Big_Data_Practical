package src;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class MatrixVectorMultiply {

    // matrix line: row,col,value
    // emit: key = col, value = M,row,value
    public static class MatrixMapper extends Mapper<Object, Text, IntWritable, Text> {
        private IntWritable outKey = new IntWritable();
        private Text outValue = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty())
                return;

            String[] parts = line.split(",");
            if (parts.length != 3)
                return;

            int row = Integer.parseInt(parts[0].trim());
            int col = Integer.parseInt(parts[1].trim());
            double val = Double.parseDouble(parts[2].trim());

            outKey.set(col);
            outValue.set("M," + row + "," + val);
            context.write(outKey, outValue);
        }
    }

    // vector line: col,value
    // emit: key = col, value = V,value
    public static class VectorMapper extends Mapper<Object, Text, IntWritable, Text> {
        private IntWritable outKey = new IntWritable();
        private Text outValue = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty())
                return;

            String[] parts = line.split(",");
            if (parts.length != 2)
                return;

            int col = Integer.parseInt(parts[0].trim());
            double val = Double.parseDouble(parts[1].trim());

            outKey.set(col);
            outValue.set("V," + val);
            context.write(outKey, outValue);
        }
    }

    // join by col, multiply matrix value with vector value
    // emit row -> partialProduct
    public static class MultiplyReducer extends Reducer<IntWritable, Text, IntWritable, DoubleWritable> {
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            double vectorVal = 0.0;
            java.util.List<String> matrixEntries = new java.util.ArrayList<>();

            for (Text t : values) {
                String s = t.toString();
                if (s.startsWith("V,")) {
                    vectorVal = Double.parseDouble(s.split(",")[1]);
                } else if (s.startsWith("M,")) {
                    matrixEntries.add(s);
                }
            }

            for (String entry : matrixEntries) {
                String[] parts = entry.split(",");
                int row = Integer.parseInt(parts[1]);
                double matrixVal = Double.parseDouble(parts[2]);
                context.write(new IntWritable(row), new DoubleWritable(matrixVal * vectorVal));
            }
        }
    }

    // second stage: sum partial products by row
    public static class SumMapper extends Mapper<Object, Text, IntWritable, DoubleWritable> {
        private IntWritable outKey = new IntWritable();
        private DoubleWritable outValue = new DoubleWritable();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().trim().split("\\t");
            if (parts.length != 2)
                return;

            outKey.set(Integer.parseInt(parts[0].trim()));
            outValue.set(Double.parseDouble(parts[1].trim()));
            context.write(outKey, outValue);
        }
    }

    public static class SumReducer extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {
        public void reduce(IntWritable key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            double sum = 0.0;
            for (DoubleWritable v : values)
                sum += v.get();
            context.write(key, new DoubleWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: MatrixVectorMultiply <matrixInput> <vectorInput> <tempOutput> <finalOutput>");
            System.exit(2);
        }

        Configuration conf = new Configuration();

        Job job1 = Job.getInstance(conf, "matrix vector multiply stage1");
        job1.setJarByClass(MatrixVectorMultiply.class);

        MultipleInputs.addInputPath(job1, new Path(args[0]), TextInputFormat.class, MatrixMapper.class);
        MultipleInputs.addInputPath(job1, new Path(args[1]), TextInputFormat.class, VectorMapper.class);

        job1.setReducerClass(MultiplyReducer.class);
        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(Text.class);
        job1.setOutputKeyClass(IntWritable.class);
        job1.setOutputValueClass(DoubleWritable.class);

        FileOutputFormat.setOutputPath(job1, new Path(args[2]));

        boolean success = job1.waitForCompletion(true);
        if (!success)
            System.exit(1);

        Job job2 = Job.getInstance(conf, "matrix vector multiply stage2");
        job2.setJarByClass(MatrixVectorMultiply.class);

        job2.setMapperClass(SumMapper.class);
        job2.setReducerClass(SumReducer.class);

        job2.setMapOutputKeyClass(IntWritable.class);
        job2.setMapOutputValueClass(DoubleWritable.class);
        job2.setOutputKeyClass(IntWritable.class);
        job2.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(job2, new Path(args[2]));
        FileOutputFormat.setOutputPath(job2, new Path(args[3]));

        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }
}