using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data;
using System.IO;

namespace Solution
{
    class Posterior
    {
        internal string Feature;
        internal int Parameter;
        internal int Label;
        internal decimal Probability;
    }


    class helper
    {
        private static int RecordCount = 0;
        private static Dictionary<int, int> priorByClass = null;
        private static List<Posterior> TrainingOrchestrator = null;

        #region Initialization and Reading Input
        internal static void Initialize(DataTable trainingData)
        {
            //calculate prior Probabilities
            RecordCount = trainingData.Rows.Count;
            int[] classes = trainingData.Select().Select(x => Convert.ToInt32(x["class_type"])).ToArray();

            //label, countOflabels
            priorByClass = classes.GroupBy(x => x)
                .ToDictionary(cls => cls.Key, cls => cls.Count());

            foreach(int c in new[] {1,2,3,4,5,6,7 })
            {
                if (!priorByClass.ContainsKey(c))
                    priorByClass.Add(c, 0);
            }
        }

        private static Tuple<DataTable,DataTable> parseInput(string[] lines)
        {
            DataColumn[] dataColumns = new DataColumn[]
            {
                new DataColumn("animal_name",typeof(string)),
                new DataColumn("hair",typeof(int)),
                new DataColumn("feathers",typeof(int)),
                new DataColumn("eggs",typeof(int)),
                new DataColumn("milk",typeof(int)),
                new DataColumn("airborne",typeof(int)),
                new DataColumn("aquatic",typeof(int)),
                new DataColumn("predator",typeof(int)),
                new DataColumn("toothed",typeof(int)),
                new DataColumn("backbone",typeof(int)),
                new DataColumn("breathes",typeof(int)),
                new DataColumn("venomous",typeof(int)),
                new DataColumn("fins",typeof(int)),
                new DataColumn("legs",typeof(int)),
                new DataColumn("tail",typeof(int)),
                new DataColumn("domestic",typeof(int)),
                new DataColumn("catsize",typeof(int)),
                new DataColumn("class_type",typeof(int))
            };

            DataTable dtTraining = new DataTable("Training");

            dtTraining.Columns.AddRange(dataColumns.ToArray());
            DataTable dtTest = dtTraining.Clone();
            dtTest.TableName = "Test";

            foreach(string line in lines)
            {
                var rowData = line.Split(',');

                if (rowData[17] == "-1")
                    dtTest.Rows.Add(rowData);
                else
                    dtTraining.Rows.Add(rowData);
            }

            return Tuple.Create(dtTraining, dtTest);
        }

        internal static Tuple<DataTable, DataTable> LoadDataFromFile(string filePath)
        {
            string[] linesTemp = System.IO.File.ReadAllLines(filePath);

            string[] lines = linesTemp.Where(x => x.Trim().Length > 0).Skip(1).ToArray();

            return parseInput(lines);
        }

        internal static Tuple<DataTable, DataTable> LoadDataFromConsole()
        {
            string stdin = null;
            if (Console.IsInputRedirected)
            {
                using (StreamReader reader = new StreamReader(Console.OpenStandardInput(), Console.InputEncoding))
                {
                    stdin = reader.ReadToEnd();
                }
            }

            string[] linesTemp = stdin.Split('\n');
            string[] lines = linesTemp.Where(x => x.Trim().Length > 0).Skip(1).ToArray();

            return parseInput(lines.Skip(1).ToArray());
        }
        
        internal static void TrainNaiveClassifier(DataTable dtTrainingData)
        {
            TrainingOrchestrator = new List<Posterior>();

            foreach(DataColumn attribute in dtTrainingData.Columns)
            {
                if (attribute.ColumnName == "animal_name" || attribute.ColumnName == "class_type") continue;
                string att = attribute.ColumnName;

                //distinct data per column
                var datarows = dtTrainingData.Select();
                
                if (att=="legs")
                {
                    foreach(int c in new[] { 1,2,3,4,5,6,7})
                    {
                        int leg0 = datarows.Where(x => Convert.ToInt32(x[att]) == 0 && Convert.ToInt32(x["class_type"]) == c).Count();
                        int leg2 = datarows.Where(x => Convert.ToInt32(x[att]) == 2 && Convert.ToInt32(x["class_type"]) == c).Count();
                        int leg4 = datarows.Where(x => Convert.ToInt32(x[att]) == 4 && Convert.ToInt32(x["class_type"]) == c).Count(); 
                        int leg5 = datarows.Where(x => Convert.ToInt32(x[att]) == 5 && Convert.ToInt32(x["class_type"]) == c).Count(); 
                        int leg6 = datarows.Where(x => Convert.ToInt32(x[att]) == 6 && Convert.ToInt32(x["class_type"]) == c).Count(); 
                        int leg8 = datarows.Where(x => Convert.ToInt32(x[att]) == 8 && Convert.ToInt32(x["class_type"]) == c).Count();

                        Posterior pLeg0 = new Posterior()
                        {
                            Feature = att,
                            Parameter = 0,
                            Label = c,
                            Probability = (leg0 + 0.1m) / (priorByClass[c] + 0.6m)
                        };
                        TrainingOrchestrator.Add(pLeg0);

                        Posterior pLeg2 = new Posterior()
                        {
                            Feature = att,
                            Parameter = 2,
                            Label = c,
                            Probability = (leg2 + 0.1m) / (priorByClass[c] + 0.6m)
                        };
                        TrainingOrchestrator.Add(pLeg2);

                        Posterior pLeg4 = new Posterior()
                        {
                            Feature = att,
                            Parameter = 4,
                            Label = c,
                            Probability = (leg4 + 0.1m) / (priorByClass[c] + 0.6m)
                        };
                        TrainingOrchestrator.Add(pLeg4);

                        Posterior pLeg5 = new Posterior()
                        {
                            Feature = att,
                            Parameter = 5,
                            Label = c,
                            Probability = (leg5 + 0.1m) / (priorByClass[c] + 0.6m)
                        };
                        TrainingOrchestrator.Add(pLeg5);

                        Posterior pLeg6 = new Posterior()
                        {
                            Feature = att,
                            Parameter = 6,
                            Label = c,
                            Probability = (leg6 + 0.1m) / (priorByClass[c] + 0.6m)
                        };
                        TrainingOrchestrator.Add(pLeg6);

                        Posterior pLeg8 = new Posterior()
                        {
                            Feature = att,
                            Parameter = 8,
                            Label = c,
                            Probability = (leg8 + 0.1m) / (priorByClass[c] + 0.6m)
                        };
                        TrainingOrchestrator.Add(pLeg8);

                    }
                }
                else
                {
                    foreach (int c in new[] { 1, 2, 3, 4, 5, 6, 7 })
                    {
                        int parm0 = datarows.Where(x => Convert.ToInt32(x[att]) == 0 && Convert.ToInt32(x["class_type"]) == c).Count();
                        int param1 = datarows.Where(x => Convert.ToInt32(x[att]) == 1 && Convert.ToInt32(x["class_type"]) == c).Count();

                        Posterior p0 = new Posterior()
                        {
                            Feature = att,
                            Parameter = 0,
                            Label = c,
                            Probability = (parm0 + 0.1m) / (priorByClass[c] + 0.2m)
                        };
                        TrainingOrchestrator.Add(p0);

                        Posterior p1 = new Posterior()
                        {
                            Feature = att,
                            Parameter = 1,
                            Label = c,
                            Probability = (param1 + 0.1m) / (priorByClass[c] + 0.2m)
                        };
                        TrainingOrchestrator.Add(p1);
                    }
                }
            }
        
        }
        #endregion Initialization and Reading Input
    
    
        internal static void RunClassifier(DataTable dtTestData)
        {
            Dictionary<string, int> InputQuery = new Dictionary<string, int>();
            foreach(DataColumn dc in dtTestData.Columns)
            {
                if (dc.ColumnName == "animal_name" || dc.ColumnName == "class_type") continue;
                InputQuery.Add(dc.ColumnName,0);
            }

            List<string> cols = InputQuery.Keys.ToList();

            foreach(DataRow dr in dtTestData.Rows)
            {

                foreach(string col in cols)
                {
                    if (InputQuery.ContainsKey(col))
                    {
                        InputQuery[col] = Convert.ToInt32(dr[col]);
                    }
                }
                // Input set ready to search.

                Dictionary<int, decimal> prob = new Dictionary<int, decimal>();
                foreach(int cls in new int[] { 1,2,3,4,5,6,7})
                {
                    decimal p = 1;
                    foreach (var x in InputQuery)
                    {
                        var x1 = x.Key;
                        var x2 = x.Value;

                        var QueryResult = TrainingOrchestrator.Where(t => t.Feature == x1 && t.Parameter == x2 && t.Label == cls).Select(r => r.Probability).ToArray();

                        foreach(decimal q in QueryResult)
                        {
                            p = p * q;
                        }
                    }

                    prob.Add(cls, p);
                }


                var max = prob.Values.Max();
                int lable = prob.Where(x => x.Value == max).Select(p => p.Key).FirstOrDefault();

                Console.WriteLine(lable);
            }
        }
    
    
    }



    class Solution
    {
        static void Main(string[] args)
        {
            //var inputSet = helper.LoadDataFromFile(@"TestData\input3.txt");
            var inputSet = helper.LoadDataFromConsole();

            DataTable trainingData = inputSet.Item1;
            DataTable testData = inputSet.Item2;

            helper.Initialize(trainingData);
            helper.TrainNaiveClassifier(trainingData);

            helper.RunClassifier(testData);
        }
    }
}
