#include <include/half_edge.hpp>
#include <Eigen/Dense>

/*
#################################################################################
#                       Vertex-related Helper Functions                         #
#################################################################################
*/

// Optinal TODO: Iterate through all neighbour vertices around the vertex
// Helpful when you implement the average degree computation of the mesh
std::vector<std::shared_ptr<Vertex>> Vertex::neighbor_vertices() {
    std::vector<std::shared_ptr<Vertex>> neighborhood;
    // Traverse all half-edges incident to this vertex
    std::shared_ptr<HalfEdge> currentEdge = this->he;
    if (currentEdge) {
        std::shared_ptr<HalfEdge> startEdge = currentEdge;
        do {
            // Add the vertex at the other end of the half-edge to the neighborhood
            neighborhood.push_back(currentEdge->twin->vertex);
            // Move to the next half-edge incident to this vertex
            currentEdge = currentEdge->twin->next;
        } while (currentEdge != startEdge);
    }
    return neighborhood; 
}


// TODO: Iterate through all half edges pointing away from the vertex
std::vector<std::shared_ptr<HalfEdge>> Vertex::neighbor_half_edges() {

    std::vector<std::shared_ptr<HalfEdge>> neighborhood;
    // Start with the half-edge incident to this vertex
    std::shared_ptr<HalfEdge> currentEdge = this->he;
    // Iterate through the incident half-edges
    if (currentEdge) {
        std::shared_ptr<HalfEdge> startEdge = currentEdge;
        do {
            //if this hf is actually deleted, it will be skipped
            if(currentEdge -> exists == false){
                currentEdge = currentEdge->twin->next;
                continue;
            }
            neighborhood.push_back(currentEdge);

            // Move to the next half-edge incident to the same vertex
            currentEdge = currentEdge->twin->next;
            // Stop if we have looped back to the start
        } while (currentEdge->id != startEdge->id);
    }
    return neighborhood;
}


// TODO: Computate quadratic error metrics coefficient, which is a 5-d vector associated with each vertex
/*
    HINT:
        Please refer to homework description about how to calculate each element in the vector.
        The final results is stored in class variable "this->qem_coff"
*/
void Vertex::compute_qem_coeff() {
    this->qem_coff = Eigen::VectorXf(5);
    // Initialize variables to store sums of quadratic terms
    this ->qem_coff[0] =  this-> neighbor_vertices().size();
    // Iterate through all neighboring vertices
    //Eigen::VectorXf neighbor_sumUp[3] = Eigen::VectorXf(3);
    float x = 0, y = 0, z = 0, sum_up = 0;
    for (const auto& neighbor : this->neighbor_vertices()) {
        Eigen::VectorXf vi= neighbor -> pos;
        sum_up +=  vi.transpose() * vi;
        x += neighbor -> pos[0];
        y += neighbor -> pos[1];
        z += neighbor -> pos[2];
    }
    this->qem_coff.segment(1, 4) << x, y, z, sum_up;
}

/*
#################################################################################
#                         Face-related Helper Functions                         #
#################################################################################
*/

// TODO: Iterate through all member vertices of the face
std::vector<std::shared_ptr<Vertex>> Face::vertices() {
    std::vector<std::shared_ptr<Vertex>> member_vertices;
    //suppose the face is based on triangle
    std::shared_ptr<HalfEdge> currentEdge = this->he;
    member_vertices.push_back(currentEdge->vertex);
    currentEdge = currentEdge->next;
    member_vertices.push_back(currentEdge -> vertex);
    currentEdge = currentEdge->next;
    member_vertices.push_back(currentEdge -> vertex);
    return member_vertices;
}


// TODO: implement this function to compute the area of the triangular face
float Face::get_area(){
    Eigen::Vector3f AB = this->he->next->vertex->pos - this->he->vertex->pos;
    Eigen::Vector3f AC = this->he->next->next->vertex->pos - this->he->vertex->pos;
    float area = 0.5 * AB.cross(AC).norm();
    return area;
}

// TODO: implement this function to compute the signed volume of the triangular face
// reference: http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf eq.(5)
float Face::get_signed_volume(){
    float volume = (1.0 / 6.0) * (this->he->vertex->pos.dot(this->he->next->vertex->pos.cross(this->he->next->next->vertex->pos)));
    return volume;
}


/*
#################################################################################
#                         Edge-related Helper Functions                         #
#################################################################################
*/

/*
    TODO: 
        Compute the contraction information for the edge (v1, v2), which will be used later to perform edge collapse
            (i) The optimal contraction target v*
            (ii) The quadratic error metrics QEM, which will become the cost of contracting this edge
        The final results is stored in class variable "this->verts_contract_pos" and "this->qem"
    Please refer to homework description for more details
*/
void Edge::compute_contraction() {
    std::shared_ptr<Vertex> v1 = this -> he ->vertex;
    std::shared_ptr<Vertex> v2 = this -> he ->twin ->vertex;
    //this->verts_contract_pos = Eigen::Vector3f(0, 0, 0);
    // calculate
    auto new_qem_coeff = v1 -> qem_coff + v2 -> qem_coff;
    this -> verts_contract_pos = new_qem_coeff.segment(1,3)/new_qem_coeff[0];
    Eigen::VectorXf right(5);
    right << (this -> verts_contract_pos.transpose())*(this -> verts_contract_pos), -2*this -> verts_contract_pos, 1;
    this -> qem =  new_qem_coeff.transpose()*right;
}


/*
    TODO: 
        Perform edge contraction functionality, which we write as (v1 ,v2) → v*, 
            (i) Moves the vertex v1 to the new position v*, remember to update all corresponding attributes,
            (ii) Connects all incident edges of v1 and v2 to v*, and remove the vertex v2,
            (iii) All faces, half edges, and edges associated with this collapse edge will be removed.
    HINT: 
        (i) Pointer reassignments
        (ii) When you want to remove mesh components, simply set their "exists" attribute to False
    Please refer to homework description for more details
*/
void Edge::edge_contraction() {
    //use v1 to store v_new
    //v1 :this->he->vertex v2:thv1 -> id == 13593 ||is->he->twin->vertex
    std::shared_ptr<Vertex> v1 = this->he->vertex;
    std::shared_ptr<Vertex> v2 = this->he->twin->vertex;
    std::shared_ptr<HalfEdge> ha = this -> he ->next -> twin -> next -> next;
    std::shared_ptr<HalfEdge> hb = this -> he -> next -> next; 
    std::shared_ptr<HalfEdge> hc = this -> he -> next -> twin -> next;
    std::shared_ptr<HalfEdge> hd = this -> he ->twin ->next->twin ->next ->next;
    std::shared_ptr<HalfEdge> hg = this -> he ->twin ->next->twin ->next;
    std::shared_ptr<HalfEdge> hf = this -> he ->twin ->next ->next;
    //update v1 pos
    v1 -> pos  = this->verts_contract_pos;
    //v1 -> he =  this ->he ->next ->next ->twin;
    //--------------delete part: 2 face, 1 vertice,  6 half edge and 3 edge:
    //2 face 
    this -> he -> face -> exists = false;
    this -> he -> twin -> face ->exists =false;
    //1 vertice
    v2 -> exists = false; 
    // 3 edge: v1v2 ,he->next he -> twin ->next
    // half edge of v1v2
    this -> he -> exists = false;
    this -> he -> twin -> exists = false;
    // he->next
    this -> he -> next -> exists = false;
    this -> he -> next -> twin -> exists = false;
    //he -> twin ->next
    this -> he ->twin ->next -> exists = false;
    this -> he -> twin -> next -> twin -> exists = false;
    //delete 3 edge
    this -> exists = false;
    this -> he ->next -> edge -> exists = false;
    this -> he ->twin -> next -> edge -> exists = false;
    //----------reassign part
    for ( auto neighbor : v2 ->neighbor_half_edges()){
        if(neighbor -> edge ->exists == true){
            neighbor ->vertex = v1;
        }
    }
    // two vertice need to be consided
    if(this ->he-> next -> twin -> vertex -> he == this ->he -> next -> twin){
        //if the he of this vertex is the deleted one 
        //assign a new one
        this ->he-> next -> twin -> vertex -> he = this -> he -> next -> next;
} 
    if(this ->he -> twin -> next -> twin -> vertex -> he == this ->he -> twin -> next -> twin){
        //if the he of this vertex is the deleted one 
        //assign a new one
        this ->he -> twin -> next -> twin -> vertex -> he = this -> he ->twin -> next -> next;
    } 
    ha -> next = hb;
    hb -> next = hc;
    hd -> next = hf;
    hf -> next = hg;
    // ----------------debug

    if(ha->face->he ->exists == false){
        ha ->face -> he = ha;
    }
    if(hg->face->he ->exists == false){
        hg ->face -> he = hg;
    }
    //-----------------end debug
    hb -> face = ha->face;
    hf -> face = hg ->face;
    v1 -> he = hc;
    v1 -> qem_coff = v1 -> qem_coff + v2 -> qem_coff;
}