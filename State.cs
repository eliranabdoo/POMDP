﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace POMDP
{
    abstract class State
    {
        public abstract IEnumerable<State> Successors(Action a);
        public abstract State Apply(Action a);
        public abstract double TransitionProbability(Action a, State sTag);
        public abstract double ObservationProbability(Action a, Observation o);
        public abstract double Reward(Action a);
        public abstract Observation RandomObservation(Action a);
    }
}
