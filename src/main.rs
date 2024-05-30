// TODO Docs
// //! This project is a Rust implementation of the AI from [a video by "Pezzza's work"](https://www.youtube.com/watch?v=EvV5Qtp_fYg&t=280s), and follows the conceptual outline given there.
// //! The physics part is highly approximate engineering.
// //! Run in release mode for better performance.
// //! This is meant to be highly optimized, this is more exploration.

use std::{collections::HashMap, f32::consts::PI, fs::File, io::BufReader, path::Path};
use ai_template::*;
use fastrand::f32;

#[derive(Clone)]
struct State {
    cart_x: f32,
    cart_x_speed: f32,
    pendulum_angle: f32,
    pendulum_ang_vel: f32,
}

impl Reinforcement for State {

    // // READ-ONLY VARIABLES

    fn num_agents() -> usize {
        1000
    }

    fn num_generations() -> usize {
        1000
    }

    fn ticks_per_evaluation() -> usize {
        60 * 10
    }

    fn tick_duration() -> f32 {
        1.0 / 60.0
    }

    fn start_nodes() -> Vec<Box<Node>> {
        vec![
            Box::new(Node::new(0.0, 0.0, identity, vec![], vec![], 0)), // Cart x            // Node 0 (input)
            Box::new(Node::new(0.0, 0.0, identity, vec![], vec![], 0)), // Pendulum x        // Node 1 (input)
            Box::new(Node::new(0.0, 0.0, identity, vec![], vec![], 0)), // Pendulum y        // Node 2 (input)
            Box::new(Node::new(0.0, 0.0, identity, vec![], vec![], 0)), // Angular velocity  // Node 3 (input)

            Box::new(Node::new(0.0, 0.0, tanh, vec![], vec![], 1)),     // Cart speed        // Node 4 (output)
        ]
    }

    fn newnode_activation_f() -> fn(f32) -> f32 {
        identity
    }


    // // ACTIONS

    fn init() -> Self {
        Self {
            cart_x: 0.0,
            cart_x_speed: 0.0,
            pendulum_angle: -PI/2.0,
            pendulum_ang_vel: 0.0,
        }
    }

    fn set_inputs(&self, dac: &mut DAC) {
        dac.nodes[0].val = self.cart_x;
        dac.nodes[1].val = self.pendulum_angle.cos();
        dac.nodes[2].val = self.pendulum_angle.sin();
        dac.nodes[3].val = self.pendulum_ang_vel;
    }

    fn get_outputs(&mut self, dac: &DAC) {
        self.cart_x_speed = dac.nodes[4].val;
    }

    fn update_physics(&mut self, delta_t: f32) {
        let new_cart_x = (self.cart_x + self.cart_x_speed * delta_t).max(-3.0).min(3.0);
        self.cart_x_speed = (new_cart_x - self.cart_x) / delta_t;
        self.cart_x = new_cart_x;

        let new_pendulum_angle = self.pendulum_angle +
            ( self.pendulum_ang_vel
            + self.pendulum_angle.sin() * self.cart_x_speed * 0.1
            - self.pendulum_angle.cos() * 0.7
            ) * delta_t;
        self.pendulum_ang_vel = (new_pendulum_angle - self.pendulum_angle) / delta_t;
        self.pendulum_angle = new_pendulum_angle;
    }

    fn update_score(&mut self, score: &mut f32, delta_t: f32) {
        let mut result: f32 = 0.0;
        if self.pendulum_angle.sin() > 0.9 {
            result += 1.0;
            if self.cart_x.abs() < 0.1 {
                result += 1.0;
            }
            if self.cart_x_speed.abs() < 0.1 {
                result += 0.2;
            }
        }
        *score += result * delta_t;
    }

    fn mutated(dac: &DAC) -> DAC {
        let f = fastrand::f32();
        let has_linked = dac.has_linked();

        if f < 0.10 && has_linked {
            dac.newnode_mutated()
        } else if f < 0.35 && has_linked {
            dac.weight_mutated()
        } else if f < 0.60 && has_linked {
            dac.bias_mutated()
        } else if f < 0.80 && dac.has_unlinked() {
            dac.newconnection_mutated()
        } else {
            dac.unmutated()
        }
    }

    fn draw_vertices() -> (Vec<InputVertex>, Option<Vec<u32>>) {
        let (vertices, indices) = load_model("./scene/shapes.obj");
        (
            vertices,
            Some(indices)
        )
    }

    fn draw_transformations(&self, matrices: &mut [Mat4; 32]) {
        println!("Cart x: {}", self.cart_x);
        matrices[0] = Mat4::translate(self.cart_x / 4.0, 0.0, 0.0);
        matrices[1] = Mat4::rotate_z(self.pendulum_angle);
        matrices[1].mult(Mat4::translate(self.cart_x / 4.0, 0.0, 0.0));
    }
    
    fn draw_view_present_mode() -> PresentMode {
        PresentMode::Fifo
    }
}

fn main() {
    let start = std::time::Instant::now();

    let mut ai: AI<State> = AI::init();
    ai.train();
    
    println!("Total: {:?}", start.elapsed()); // Longer esp. with printing operations.

    let mut vk: Vk = Vk::init::<State>();
    vk.view_agent(ai.best_agent());
}

// Code from a vulkanalia tutorial: https://github.com/KyleMayes/vulkanalia/blob/master/tutorial/src/27_model_loading.rs
fn load_model<P: AsRef<Path>>(path: P) -> (Vec<InputVertex>, Vec<u32>) {
    // Model

    let mut reader = BufReader::new(File::open(path).unwrap());

    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            single_index: false,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        },
        |_| Ok(Default::default()),
    ).unwrap();

    // Vertices / Indices

    let mut unique_vertices = HashMap::new();

    let (mut vertices, mut indices): (Vec<InputVertex>, Vec<u32>) = (Vec::new(), Vec::new());

    let mut z: f32 = -0.1;
    for model in &models {
        z+=0.1;
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;

            let vertex: InputVertex = if -model.mesh.positions[pos_offset + 2] < 0.1 {
                InputVertex::new(
                    [(f32() + 4.0) / 8.0, 0.0, (f32() + 4.0) / 8.0, 1.0],
                    [
                        (model.mesh.positions[pos_offset]) / 2.0,
                        (- model.mesh.positions[pos_offset + 2]) / 2.0,
                        z,
                    ],
                    0,
                )
                    
            } else if -model.mesh.positions[pos_offset + 2] < 0.3 {
                InputVertex::new(
                    [0.0, (f32() + 4.0) / 8.0, 0.0, 1.0],
                    [
                        (model.mesh.positions[pos_offset]) / 2.0,
                        (- model.mesh.positions[pos_offset + 2] - 0.2) / 2.0,
                        z,
                    ],
                    1,
                )
                    
            } else {
                InputVertex::new(
                    [1.0, 1.0, 1.0, 1.0],
                    [
                        (model.mesh.positions[pos_offset]) / 2.0,
                        (- model.mesh.positions[pos_offset + 2]) / 2.0,
                        z,
                    ],
                    2,
                )
                    
            };

            if let Some(index) = unique_vertices.get(&vertex) {
                indices.push(*index as u32);
            } else {
                let index = vertices.len();
                unique_vertices.insert(vertex, index);
                vertices.push(vertex);
                indices.push(index as u32);
            }
        }
    }

    (vertices, indices)
}