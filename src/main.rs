//! This project is a Rust implementation of the AI from [a video by "Pezzza's work"](https://www.youtube.com/watch?v=EvV5Qtp_fYg&t=280s), and follows the conceptual outline given there.
//! The physics part is highly approximate engineering.
//! Run in release mode for better performance.
//! This is not really optimized, this is more exploration.

use std::f32::consts::PI;

use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

const AGENTS_NUM: usize = 2000;
const NUM_GENERATIONS: usize = 200;
const TICKS_PER_EVALUATION: usize = 60 * 100; // 100 seconds
const TICK_DURATION: f32 = 1.0 / 60.0; // 60 modifications per second
const GRAVITY: f32 = 1.0;

/// Top-level of the training process 
pub struct AI {
    agents: Vec<Agent>,
    past_agents: Vec<Vec<Agent>>,
}

#[derive(Clone)]
struct Agent {
    dac: DAC,
    score: f32,
    
    instant: f32,

    cart_x: f32,
    cart_x_speed: f32,
    pendulum_angle: f32,
    pendulum_ang_vel: f32,
}

/// Directed Acyclic Graph, the neural network
#[derive(Clone, Debug)]
struct DAC {
    nodes: Vec<Box<Node>>,
    order: Vec<usize>,
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
        (0..NUM_GENERATIONS).into_iter().for_each(|gen| {
            println!("Testing generation {}...", gen);
            self.evaluate();
            self.selectmutate();
        })
    }

    pub fn render_best(&self) {
        extern crate glutin_window;
        extern crate graphics;
        extern crate opengl_graphics;
        extern crate piston;
        
        use glutin_window::GlutinWindow as Window;
        use opengl_graphics::{GlGraphics, OpenGL};
        use piston::event_loop::{EventSettings, Events};
        use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
        use piston::window::WindowSettings;
        
        pub struct View {
            gl: GlGraphics, // OpenGL drawing backend.
        }
        
        impl View {
            fn render(&mut self, args: &RenderArgs, cart_x: f64, pendulum_angle: f64) {
                use graphics::*;
        
                const DARK: [f32; 4] = [0.1, 0.15, 0.3, 1.0];
                const CART: [f32; 4] = [0.8, 0.6, 0.2, 1.0];
                const PENDULUM: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        
                let cart = rectangle::rectangle_by_corners(-50.0, -5.0, 50.0, 5.0);
                let pendulum_rod = rectangle::rectangle_by_corners(0.0, -2.5, 90.0, 2.5);
                let pendulum_ball = rectangle::rectangle_by_corners(90.0, -10.0, 110.0, 10.0);
                let (x, y) = (args.window_size[0] / 2.0, args.window_size[1] / 2.0);
        
                self.gl.draw(args.viewport(), |c, gl| {
                    // Clear the screen.
                    clear(DARK, gl);
        
                    let cart_transform = c
                        .transform
                        .trans(x + cart_x, y);
                    let pendulum_transform: [[f64; 3]; 2] = c
                        .transform
                        .trans(x + cart_x, y)
                        .rot_rad(pendulum_angle);
        
                    rectangle(CART, cart, cart_transform, gl);
                    rectangle(PENDULUM, pendulum_rod, pendulum_transform, gl);
                    ellipse(PENDULUM, pendulum_ball, pendulum_transform, gl);
                });
            }
        
            fn update(&mut self, _args: &UpdateArgs) {
        
            }
        }


        
        println!("Best DAC score: {}.\nDisplaying the evaluation...", self.past_agents
            .iter().last().unwrap()
            .iter().nth(0).unwrap()
            .score
        );
        let best_dac = self.past_agents
            .iter().last().unwrap()
            .iter().nth(0).unwrap()
            .dac.clone();
        println!("Best DAC network: {:?}", best_dac);
        // let best_dac = self.agents
        //     .iter().last().unwrap()
        //     .dac.clone();
        let mut agent = Agent {
            dac: best_dac,
            score: 0.1,

            instant: 0.0,

            cart_x: 0.0,
            cart_x_speed: 0.0,
            pendulum_angle: -PI/2.0,
            pendulum_ang_vel: 0.0,
        };


        // Running the visual simulation.
        let opengl = OpenGL::V3_2;
        let mut window: Window = WindowSettings::new("Pendulum Simulation", [200, 200])
            .graphics_api(opengl)
            .exit_on_esc(true)
            .samples(8)
            .build()
            .unwrap();
        let mut view = View {
            gl: GlGraphics::new(opengl),
        };
        let mut events = Events::new(EventSettings {
                max_fps: 60,
                ups: 60,
                swap_buffers: true,
                bench_mode: false,
                lazy: false,
                ups_reset: 2,
            });
        let mut frame_count: usize = 0;
        while let Some(e) = events.next(&mut window) {
            if frame_count > TICKS_PER_EVALUATION {
                break;
            }
            if let Some(args) = e.render_args() {
                agent.evaluate_step();
                view.render(&args, agent.cart_x as f64 * 100.0, (2.0 * PI - agent.pendulum_angle) as f64);
            }
    
            if let Some(args) = e.update_args() {
                frame_count += 1;
                view.update(&args);
            }
        }
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
            score: 0.1,

            instant: 0.0,

            cart_x: 0.0,
            cart_x_speed: 0.0,
            pendulum_angle: -PI/2.0,
            pendulum_ang_vel: 0.0,
        }
    }

    fn modified(&self) -> Self {
        Self {
            dac: self.dac.modified(),
            score: 0.1,

            instant: 0.0,

            cart_x: 0.0,
            cart_x_speed: 0.0,
            pendulum_angle: -PI/2.0,
            pendulum_ang_vel: 0.0,
        }
    }

    fn evaluate(&mut self) {
        // Run network, Update physics values (cart speed, cart pos, pendulum physics data from new position), then run again, for a number of ticks and a given tick duration.
        self.dac.reordered();
        for _ in 0..TICKS_PER_EVALUATION {
            self.evaluate_step();
        }
    }

    fn evaluate_step(&mut self) {
        self.dac.run();
        self.cart_x_speed = self.dac.nodes[4].val;

        self.update_physics();
        // Modify network inputs
        self.dac.nodes[0].val = self.cart_x;
        self.dac.nodes[1].val = self.pendulum_angle.cos();
        self.dac.nodes[2].val = self.pendulum_angle.sin();
        self.dac.nodes[3].val = self.pendulum_ang_vel;

        if self.pendulum_angle.sin() > 0.9 {
            self.score += 1.0;
        }

        self.instant += TICK_DURATION;
    }

    fn update_physics(&mut self) {
        // New angle - Previous angle => ang vel
        self.cart_x += self.cart_x_speed * TICK_DURATION;
        if self.cart_x > 3.0 {
            self.cart_x = 3.0;
            self.cart_x_speed = 0.0;
        }
        else if self.cart_x  < -3.0 {
            self.cart_x = -3.0;
            self.cart_x_speed = 0.0;
        }

        self.pendulum_ang_vel += 
            - self.pendulum_angle.cos() * GRAVITY * GRAVITY * TICK_DURATION // Gravity
            + self.pendulum_angle.sin() * 3.0 * self.cart_x_speed * TICK_DURATION; // Moving cart
        self.pendulum_angle += self.pendulum_ang_vel * TICK_DURATION;
    }
}

impl DAC {
    /// The constructor for the pendulum network in [Pezzza's video](https://www.youtube.com/watch?v=EvV5Qtp_fYg)
    fn pezzzas_pendulum() -> Self {
        Self {
            nodes: vec![
                Box::new(Node::new(0.0, fastrand::f32() - 0.5, identity, vec![], vec![], 0)), // Cart x
                Box::new(Node::new(0.0, fastrand::f32() - 0.5, identity, vec![], vec![], 0)), // Pendulum x
                Box::new(Node::new(0.0, fastrand::f32() - 0.5, identity, vec![], vec![], 0)), // Pendulum y
                Box::new(Node::new(0.0, fastrand::f32() - 0.5, identity, vec![], vec![], 0)), // Angular velocity

                Box::new(Node::new(0.0, fastrand::f32() - 0.5, tanh, vec![], vec![], 1)),
            ],
            order: vec![],
        }
    }

    /// Returns a node processing order, such that children are always processed aftertheir parents.
    fn reordered(&mut self) {
        // let start = std::time::Instant::now();

        let len = self.nodes.len();
        self.order = Vec::with_capacity(len);
        let mut active_nodes: Vec<bool> = vec![true; len];
        
        while self.order.len() != len {
            // println!("Starting nodes: {:?}", self.nodes);
            // println!("Reordered nodes: {:?}", reordered);
            // println!("Active nodes: {:?}", active_nodes);
            let active_nodes_clone = active_nodes.clone();
            self.nodes.iter().enumerate()
                // Filter ---------
                .zip(active_nodes.iter_mut())
                .filter(|(_, kept)| **kept)
                // ----------------
                .for_each(|((node_i, node), kept)| {
                    if node.parents.iter().all(|parent| !active_nodes_clone[*parent]) {
                        self.order.push(node_i);
                        *kept = false;
                    }
                });
        }
        // println!("Reordered in: {}", start.elapsed().as_secs_f32());
        // println!("Reordered");
    }

    fn run(&mut self) {
        // let start = std::time::Instant::now();

        let mut vals: Vec<f32> = self.nodes.iter()
            .map(|node| node.val)
            .collect();
        for index in self.order.iter() {
            self.nodes[*index].val = vals[*index];
            self.nodes[*index].update();
            self.nodes[*index].children.iter().for_each(|(child_i, conn_weight)| {
                vals[*child_i] += self.nodes[*index].val * conn_weight;
            })
        }

        // println!("{:?}", self.nodes);

        // println!("Ran in: {}", start.elapsed().as_secs_f32());
    }

    fn modified(&self) -> Self {
        let f = fastrand::f32();
        // println!("Modifying");
        if f < 0.10 && !self.nodes.iter().all(|node| node.children.is_empty() && node.parents.is_empty()) {
            // println!("Update newnode");
            self.newnode_modified()
        } else if f < 0.30 && self.nodes.iter().any(|parent_node| parent_node.children.len() < self.nodes.iter().filter(|node| node.layer > parent_node.layer).count()) {
            // println!("Update newconnection");
            self.newconnection_modified()
        } else if f < 0.80 && self.nodes.iter().any(|node| !node.children.is_empty()){
            // println!("Update weight");
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
            fastrand::u32((new.nodes[connection_parent_i].layer+1)..(new.nodes[connection_child_i].layer))
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
        // println!("{}", self.val);
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
    let start = std::time::Instant::now();

    let mut ai = AI::init();
    ai.train();
    
    println!("Total: {}", start.elapsed().as_secs_f32()); // Longer esp. with printing operations.
    
    ai.render_best();
}