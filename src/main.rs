//! This project is a Rust implementation of the AI from [a video by "Pezzza's work"](https://www.youtube.com/watch?v=EvV5Qtp_fYg&t=280s), and follows the conceptual outline given there.
//! Run in release mode for better performance.
//! Still, this is not really optimized, this is more exploration.

use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

const AGENTS_NUM: usize = 1000;
const NUM_GENERATIONS: usize = 200;
const TICKS_PER_EVALUATION: usize = 60 * 100; // 100 seconds
const TICK_DURATION: f32 = 1.0 / 60.0; // 60 modifications per second

/// Top-level of the training process 
pub struct AI {
    agents: Vec<Agent>,
    past_agents: Vec<Vec<Agent>>,
}

#[derive(Clone)]
struct Agent {
    dac: DAC,

    score: f32,

    cart_speed: f32,

    instant: f32,
    cart_x: f32,
    pendulum_dir_x: f32,
    pendulum_dir_y: f32,
    pendulum_ang_vel: f32,
}

/// Directed Acyclic Graph, the neural network
#[derive(Clone, Debug)]
struct DAC {
    nodes: Vec<Box<Node>>,
}

/// A neuron
#[derive(Clone, Debug)]
struct Node {
    val: f32,
    bias: f32,
    activation_f: fn(f32) -> f32,
    /// Parents indices
    parents: Vec<usize>,
    /// Children (indices, connections weights)
    children: Vec<(usize, f32)>,
    /// The layer of this node
    layer: u32,
}

impl AI {
    pub fn init() -> Self {
        Self {
            agents: (0..AGENTS_NUM).map(|_| Agent::new()).collect(),
            past_agents: Vec::new(),
        }
    }

    pub fn train(&mut self) {
        (0..NUM_GENERATIONS).into_iter().for_each(|_| {
            self.evaluate();
            self.selectmutate();
        })
    }

    fn evaluate(&mut self) {
        self.agents.par_iter_mut().for_each(|agent| {
            agent.evaluate();
        })
    }

    fn selectmutate(&mut self) {
        self.agents.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap()); // Bigger scores come before

        let mut cumulative_sum: Vec<f32> = Vec::with_capacity(AGENTS_NUM);
        let mut total_score = 0.0;
        self.agents.iter()
            .map(|agent| agent.score)
            .for_each(|score| {
                total_score+=score;
                cumulative_sum.push(total_score);
            });
        cumulative_sum.iter_mut()
            .for_each(|score| {
                *score/=total_score;
            });

        let mut new_agents = self.agents[0..AGENTS_NUM/3].to_vec();
        while new_agents.len() < AGENTS_NUM {
            let random = fastrand::f32();
            
            new_agents.push(self.agents[
                cumulative_sum.iter().position(|&cumulative_score| cumulative_score > random).unwrap()
            ].modified());
        }

        std::mem::swap(&mut self.agents, &mut new_agents);
        self.past_agents.push(new_agents); // `new_agents` are the old ones, as we just swapped them.
    }
}

impl Agent {
    fn new() -> Self {
        Self {
            dac: DAC::pezzzas_pendulum(),
        
            score: 0.0,
        
            cart_speed: 0.0,
        
            instant: 0.0,
            cart_x: 0.0,
            pendulum_dir_x: 0.0,
            pendulum_dir_y: -1.0,
            pendulum_ang_vel: 0.0,
        }
    }

    fn evaluate(&mut self) {
        // Run network, Update physics values (cart speed, cart pos, pendulum physics data from new position), then run again, for a number of ticks and a given tick duration.
        // self.instant = 0.0;
        for _ in 0..TICKS_PER_EVALUATION {
            self.dac.run();
            self.cart_speed = self.dac.nodes[4].val;
            self.update_physics();

            if self.pendulum_dir_y > 0.8 {
                self.score += 1.0;
            }

            // self.instant += TICK_DURATION;
        }
    }

    fn update_physics(&mut self) {
        todo!("Physics update")
    }

    fn modified(&self) -> Self {
        Self {
            dac: self.dac.modified().reordered(),
        
            score: 0.0,
        
            cart_speed: 0.0,
        
            instant: 0.0,
            cart_x: 0.0,
            pendulum_dir_x: 0.0,
            pendulum_dir_y: -1.0,
            pendulum_ang_vel: 0.0,
        }
    }
}

impl DAC {
    /// The constructor for the pendulum network in [Pezzza's video](https://www.youtube.com/watch?v=EvV5Qtp_fYg)
    fn pezzzas_pendulum() -> Self {
        Self {
            nodes: vec![
                Box::new(Node::new(0.0, fastrand::f32() - 0.5, identity, vec![], vec![], 0)),
                Box::new(Node::new(0.0, fastrand::f32() - 0.5, identity, vec![], vec![], 0)),
                Box::new(Node::new(0.0, fastrand::f32() - 0.5, identity, vec![], vec![], 0)),
                Box::new(Node::new(0.0, fastrand::f32() - 0.5, identity, vec![], vec![], 0)),

                Box::new(Node::new(0.0, fastrand::f32() - 0.5, tanh, vec![], vec![], 1)),
            ],
        }
    }

    /// Returns reordered nodes, such that children are always processed aftertheir parents.
    fn reordered(mut self) -> Self {
        #[cfg(debug_assertions)]
        let start = std::time::Instant::now();

        let len = self.nodes.len();
        let mut reordered: Vec<Box<Node>> = Vec::with_capacity(len);
        let mut active_nodes: Vec<bool> = vec![true; len];

        while reordered.len() != len {
            let active_nodes_clone = active_nodes.clone();
            self.nodes.iter()
                // Filter ---------
                .zip(active_nodes.iter_mut())
                .filter(|(_, kept)| **kept)
                // ----------------
                .for_each(|(node, kept)| {
                    if node.parents.iter().all(|parent| !active_nodes_clone[*parent]) {
                        reordered.push((*node).clone());
                        *kept = false;
                    }
                });
        }
        self.nodes = reordered;
        #[cfg(debug_assertions)]
        println!("Reordered in: {}", start.elapsed().as_secs_f32());

        self
    }

    fn run(&mut self) {
        #[cfg(debug_assertions)]
        let start = std::time::Instant::now();

        let mut vals: Vec<f32> = self.nodes.iter()
            .map(|node| node.val)
            .collect();
        for (node_i, node) in self.nodes.iter_mut().enumerate() {
            node.val = vals[node_i];
            node.update();
            node.children.iter().for_each(|(child_i, conn_weight)| {
                vals[*child_i] += node.val * conn_weight;
            })
        }

        #[cfg(debug_assertions)]
        println!("{:?}", self.nodes);

        #[cfg(debug_assertions)]
        println!("Ran in: {}", start.elapsed().as_secs_f32());
    }

    fn modified(&self) -> Self {
        let f = fastrand::f32();
        if f < 0.10 && !self.nodes.iter().all(|node| node.children.is_empty() && node.parents.is_empty()) {
            self.newnode_modified()
        } else if f < 0.20 && self.nodes.iter().any(|parent_node| parent_node.children.len() < self.nodes.iter().filter(|node| node.layer > parent_node.layer).count()) {
            self.newconnection_modified()
        } else if f < 0.80 {
            self.weight_modified()
        } else { // Nothing
            self.unmodified()
        }
    }

    fn newnode_modified(&self) -> Self {
        let mut new = self.clone();

        let connection_parent_i: usize;
        let connection_child_i: usize;
        let connection_weight: f32;
        let nodes_len = new.nodes.len();
        loop {
            let parents_first = fastrand::bool();
            let node_i = fastrand::usize(0..nodes_len);
            if parents_first {
                if !new.nodes[node_i].parents.is_empty() {
                    connection_child_i = node_i;
                    connection_parent_i = new.nodes[node_i].parents[fastrand::usize(0..new.nodes[node_i].parents.len())];
                    connection_weight = new.nodes[connection_parent_i].children.iter().filter(|child| child.0 == connection_child_i).collect::<Vec<&(usize, f32)>>()[0].1;
                    break;
                } else {
                    if !new.nodes[node_i].children.is_empty() {
                        connection_parent_i = node_i;
                        (connection_child_i, connection_weight) = new.nodes[node_i].children[fastrand::usize(0..new.nodes[node_i].children.len())];
                        break;
                    }
                }
            } else {
                if !new.nodes[node_i].children.is_empty() {
                    connection_parent_i = node_i;
                    (connection_child_i, connection_weight) = new.nodes[node_i].children[fastrand::usize(0..new.nodes[node_i].children.len())];
                    break;
                } else {
                    if !new.nodes[node_i].children.is_empty() {
                        connection_child_i = node_i;
                        connection_parent_i = new.nodes[node_i].parents[fastrand::usize(0..new.nodes[node_i].parents.len())];
                        connection_weight = new.nodes[connection_parent_i].children.iter().filter(|child| child.0 == connection_child_i).collect::<Vec<&(usize, f32)>>()[0].1;
                        break;
                    }
                }
            }
        }

        // Update parent
        let parent_inner_i = new.nodes[connection_child_i].parents.iter().position(|&parent| parent == connection_parent_i).unwrap();
        new.nodes[connection_child_i].parents.remove(parent_inner_i);
        new.nodes[connection_child_i].parents.push(nodes_len);

        // Update child
        let child_inner_i = new.nodes[connection_parent_i].children.iter().position(|&(child, _)| child == connection_child_i).unwrap();
        new.nodes[connection_parent_i].children.remove(child_inner_i);
        new.nodes[connection_parent_i].children.push((nodes_len, connection_weight));

        let mut update_layers = false;
        let new_node_layer: u32 = if new.nodes[connection_child_i].layer - new.nodes[connection_parent_i].layer == 1 {
            update_layers = true;
            new.nodes[connection_child_i].layer
        } else {
            fastrand::u32((new.nodes[connection_parent_i].layer+1)..(new.nodes[connection_child_i].layer-1))
        };

        if update_layers {
            new.nodes.iter_mut()
                .filter(|node| node.layer >= new_node_layer)
                .for_each(|node| node.layer += 1);
        }
        
        new.nodes.push(Box::new(Node::new(0.0, fastrand::f32() - 0.5, relu, vec![connection_parent_i], vec![(connection_child_i, connection_weight)], new_node_layer)));
        new
    }

    fn newconnection_modified(&self) -> Self {
        let mut new = self.clone();

        let node_a_i: usize = fastrand::usize(0..new.nodes.len());
        let node_b_i: usize;
        loop {
            let b_i = fastrand::usize(0..new.nodes.len());
            if new.nodes[b_i].layer != new.nodes[node_a_i].layer {
                node_b_i = b_i;
                break;
            }
        }

        let a_is_child = new.nodes[node_a_i].layer > new.nodes[node_b_i].layer;
        if a_is_child {
            new.nodes[node_a_i].parents.push(node_b_i);
            new.nodes[node_b_i].children.push((node_a_i, 0.5));
        } else {
            new.nodes[node_b_i].parents.push(node_a_i);
            new.nodes[node_a_i].children.push((node_b_i, 0.5));
        }

        new
    }

    fn weight_modified(&self) -> Self {
        let mut new = self.clone();
        
        let children_count = new.nodes.iter()
            .flat_map(|node| node.children.iter())
            .count();
        let index = fastrand::usize(0..children_count);
        new.nodes.iter_mut()
            .flat_map(|node| node.children.iter_mut())
            .nth(index).unwrap().1 += fastrand::f32() - 0.5;

        new
    }

    fn unmodified(&self) -> Self {
        self.clone()
    }
}

impl Node {
    /// A new input node.
    fn new(val: f32, bias: f32, activation_f: fn(f32) -> f32, parents: Vec<usize>, children: Vec<(usize, f32)>, layer: u32) -> Self {
        Self {
            val,
            bias,
            activation_f,
            parents,
            children,
            layer,
        }
    }

    /// Updates `val` from `bias` and `activation_f`.
    fn update(&mut self) {
        self.val = (self.activation_f)(self.val + self.bias);
        #[cfg(debug_assertions)]
        println!("{}", self.val);
    }
}

/// The identity function
fn identity(input: f32) -> f32 {
    input
}

/// The hyperbolic tangent function
fn tanh(input: f32) -> f32 {
    input.tanh()
}

/// The REctified Linear Unit function
fn relu(input: f32) -> f32 {
    input.max(0.0)
}

fn main() {
    unimplemented!("Not ready yet!");

    // let start = std::time::Instant::now();

    // let ai = AI::init();

    // println!("Total: {}", start.elapsed().as_secs_f32()); // Longer esp. with printing operations.
}