/// Directed Acyclic Graph
#[derive(Debug)]
struct DAC {
    nodes: Vec<Box<Node>>,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    pub val: f32,
    bias: f32,
    activation_f: fn(f32) -> f32,
    /// Parents indices
    parents: &'static [usize],
    /// Children (indices, connections weights)
    children: &'static [(usize, f32)],
}

impl DAC {
    /// The constructor for the network in [Pezzza's video](https://www.youtube.com/watch?v=EvV5Qtp_fYg)
    fn pezzzas() -> DAC {
        DAC {
            nodes: vec![
                Box::new(Node::new_input(&[(6, 0.5), (8, 0.33)])),  // Node 0
                Box::new(Node::new_input(&[(8, 0.33)])),            // Node 1
                Box::new(Node::new_input(&[(9, 0.33)])),            // etc.
                Box::new(Node::new_input(&[(8, 0.33), (9, 0.33)])),
                Box::new(Node::new_output(&[6, 7, 9])),
                Box::new(Node::new_output(&[7, 9])),
                Box::new(Node::new_hidden (&[0, 8], &[(4, 0.33)])),
                Box::new(Node::new_hidden (&[8], &[(4, 0.33), (5, 0.5)])),
                Box::new(Node::new_hidden (&[0, 1, 3], &[(6, 0.5), (7, 1.0), (9, 0.33)])),
                Box::new(Node::new_hidden (&[2, 3, 8], &[(4, 0.33), (5, 0.5)])),
            ],
        }
    }

    /// Reorders nodes, such that children are always processed aftertheir parents.
    /// Returns the *indices* of the nodes to be processed, in order.
    fn reorder(&mut self) {
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
}

impl Node {
    /// A new input node.
    fn new_input(children: &'static [(usize, f32)]) -> Self {
        Self {
            val: 0.0,
            bias: fastrand::f32(),
            activation_f: identity,
            parents: &[],
            children,
        }
    }
    /// A new hidden node.
    fn new_hidden(parents: &'static [usize], children: &'static [(usize, f32)]) -> Self {
        Self {
            val: 0.0,
            bias: fastrand::f32(),
            activation_f: tanh,
            parents,
            children,
        }
    }
    /// A new output node.
    fn new_output(parents: &'static [usize]) -> Self {
        Self {
            val: 0.0,
            bias: fastrand::f32(),
            activation_f: tanh,
            parents,
            children: &[],
        }
    }

    /// Updates `val` from `bias` and `activaiton_f`.
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

fn main() {
    let start = std::time::Instant::now();

    let mut model: DAC = DAC::pezzzas();

    model.reorder();
    model.nodes[0].val =  0.37;
    model.nodes[1].val =  0.72;
    model.nodes[2].val = -0.52;
    model.nodes[3].val = -0.89;
    model.run();

    println!("Total: {}", start.elapsed().as_secs_f32()); // Longer esp. with printing operations.
}