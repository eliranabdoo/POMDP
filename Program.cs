using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Diagnostics;
using System.IO;

namespace POMDP
{
    class Program
    {
        static void Main(string[] args)
        {
            FileStream fs = new FileStream("Debug.txt", FileMode.Create);
            Debug.Listeners.Add(new TextWriterTraceListener(Console.Out));
            Debug.Listeners.Add(new TextWriterTraceListener(fs));

            MazeDomain maze = new MazeDomain("Maze2.txt");

            PointBasedValueIteration pbvi = new PointBasedValueIteration(maze);
            pbvi.PointBasedVI(100, 25);

            //MDPValueFunction v = new MDPValueFunction(maze);
            //v.ValueIteration(0.5);

            //MostLikelyStatePolicy p1 = new MostLikelyStatePolicy(v);
            //VotingPolicy p2 = new VotingPolicy(v);
            //QMDPPolicy p3 = new QMDPPolicy(v, maze);

            //double dADR1 = maze.ComputeAverageDiscountedReward(p1, 100, 100);
            //double dADR2 = maze.ComputeAverageDiscountedReward(p2, 100, 100);
            //double dADR3 = maze.ComputeAverageDiscountedReward(p3, 100, 100);
            double dADR4 = maze.ComputeAverageDiscountedReward(pbvi, 100, 100);

            MazeViewer viewer = new MazeViewer(maze);
            viewer.Start();
            maze.SimulatePolicy(pbvi, 100, viewer);

            Debug.Close();
        }
    }
}
