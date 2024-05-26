// TODO Docs
// //! This project is a Rust implementation of the AI from [a video by "Pezzza's work"](https://www.youtube.com/watch?v=EvV5Qtp_fYg&t=280s), and follows the conceptual outline given there.
// //! The physics part is highly approximate engineering.
// //! Run in release mode for better performance.
// //! This is meant to be highly optimized, this is more exploration.

use std::f32::consts::PI;
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
        100
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

    fn update_physics(&mut self) {
        let new_cart_x = (self.cart_x + self.cart_x_speed * Self::tick_duration()).max(-3.0).min(3.0);
        self.cart_x_speed = (new_cart_x - self.cart_x) / Self::tick_duration();
        self.cart_x = new_cart_x;

        let new_pendulum_angle = self.pendulum_angle +
            ( self.pendulum_ang_vel
            + self.pendulum_angle.sin() * self.cart_x_speed * 0.1
            - self.pendulum_angle.cos() * 0.7
            ) * Self::tick_duration();
        self.pendulum_ang_vel = (new_pendulum_angle - self.pendulum_angle) / Self::tick_duration();
        self.pendulum_angle = new_pendulum_angle;
    }

    fn update_score(&mut self, score: &mut f32) {
        if self.pendulum_angle.sin() > 0.9 {
            *score += 1.0;
            if self.cart_x.abs() < 0.1 {
                *score += 1.0;
            }
            if self.cart_x_speed.abs() < 0.1 {
                *score += 0.2;
            }
        }
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
        use tobj;

        let cart = tobj::load_obj("./scene/cart.obj", &tobj::GPU_LOAD_OPTIONS);
        assert!(cart.is_ok());
        
        let (cart_models, cart_materials) = cart.expect("Failed to load OBJ file");
        
        // Materials might report a separate loading error if the MTL file wasn't found.
        // If you don't need the materials, you can generate a default here and use that
        // instead.
        let cart_materials = cart_materials.expect("Failed to load MTL file");
        
        println!("# of models: {}", cart_models.len());
        println!("# of materials: {}", cart_materials.len());
        
        let cart_indices = cart_models.iter()
            .flat_map(|model| model.mesh.indices.clone())
            .collect::<Vec<u32>>();
        let cart_vertices = cart_models.iter()
            .flat_map(|model| model.mesh.positions.chunks_exact(3).collect::<Vec<&[f32]>>())
            .map(|chunk| InputVertex::new(
                [f32(), 1.0, f32(), 1.0],
                [chunk[0] * 5.0, chunk[2] * 5.0, 0.0],
                0
            ))
            .collect::<Vec<InputVertex>>();
        (
            cart_vertices,
            Some(cart_indices)
        )
    }

    fn draw_transformations(&self, matrices: &mut [Mat4; 32]) {
        matrices[0] = Mat4::translate(0.5, 0.0, 0.0);
        matrices[1] = Mat4::translate(0.0, 0.5, 0.0);
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

    let mut vk: Vk<State> = Vk::init();
    vk.view();
}