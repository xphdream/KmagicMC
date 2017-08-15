#include<iostream>
#include<fstream>
#include<istream>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<vector>
#include<stdlib.h>
#include<stdio.h>
#include<stdarg.h> //Added by John
#include<limits.h>
#include<fnmatch.h>
#include<sys/types.h>
#include<unistd.h>
#include<dirent.h>
#include<iomanip>
#include<time.h>
#include "Array.h"
#include<map>
#include<bitset>  
#include<unordered_map>
#include <time.h> 
#include <list> 
#include <xxhash32.h>
#include <array>

using namespace std;

//NOTE: This is licensed software. See LICENSE.txt for license terms.
//written by Anton Van der Ven, John Thomas, Qingchuan Xu, and Jishnu Bhattacharya
//please see CASMdocumentation.pdf for a tutorial for using the code.
//Version as of May 26, 2010

/*Changes since clusters10.0.h
  ====================
  x  Average correlations for MC simulation
  x  Fix int_to_string and add double_to_string
  x  Generalized Susceptibility for multiple sublattices
  x  Symmetry classification (currently only output for point_group)
  x  Tensor Class
  x  read_mc_input routine
*/
//**************************************************************
//**************************************************************
double tol = 1.0e-3;
double kb  = 0.000086173324; // eV/K

// xph: total number of moves for each site.
const int nmoves=14;
// xph: barriers
// Li and Ni have different Ea0 (EKRA)
// todo: Ea0 should be read from input file
double const Ea0_Ni = 0.50;
double const Ea0_Li = 0.15;
double const Ea0_Li_o2o = 0.55; // v1.6: Li inlayer octahedral to octahedral hop
double const freq = 1.0e+7; //in the unit of 1/(mu_s), which makes time unit being mu_s (10e-6 s)

////////////////////////////////////////////////////////////////////////////////
//swoboda
char basis_type;
////////////////////////////////////////////////////////////////////////////////
void coord_trans_mat(double lat[3][3], double FtoC[3][3], double CtoF[3][3]);


class vec;
class sym_op;
class tensor;
class specie;
class atompos;
class cluster;
class orbit;
class multiplet;
class structure;
class concentration;
class arrangement;
class superstructure;
class configurations;
class facet;
class hull;
class chempot;
class trajectory;
class fluctuation;
class hop;
class mc_index;
class Monte_Carlo;


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class vec{
 public:
  bool frac_on,cart_on;
  double fcoord[3],ccoord[3];
  double length;

  vec() {frac_on=false; cart_on=false;}

  double calc_dist();
  vec apply_sym(sym_op op);
  void print_frac(ostream &stream);
  void print_cart(ostream &stream);
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class sym_op{
 public:

  bool frac_on,cart_on;
  double fsym_mat[3][3],csym_mat[3][3];
  double ftau[3],ctau[3];
  double lat[3][3];
  double FtoC[3][3],CtoF[3][3];
  vec eigenvec; //Added by John to hold rotation axis for rotation or mirror plane normal
  short int sym_type;  //Added by John, Not calculated=-2, Inversion=-1, Identity=0, Rotation=1, Mirror=2, Rotoinversion=3
  short int op_angle; //Added by John to hold rotation angle for rotation op


  sym_op(){frac_on=false; cart_on=false;}

  void get_sym_type(); //Added by John, populates eigenvec and sym_type;
  void print_fsym_mat(ostream &stream);
  void print_csym_mat(ostream &stream);
  void get_trans_mat(){ coord_trans_mat(lat,FtoC,CtoF); }
  void get_csym_mat();
  void get_fsym_mat();
  void update();
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//Edited by John
//generalized for any tensor, though needs to be cleaned up.  Some constructors may be redundant.
// ?*?*? May decide to replace unsized arrays with vectors for stability.

class tensor{
 public:
  int rank, size;
  int *dim, *mult;
  double *K;
  tensor();  //Default constructor
  tensor(tensor const& ttens);  //Copy constructor
  tensor& operator=(const tensor& ttens);  //Assignment operator
  ~tensor(){delete [] dim; delete [] mult; delete [] K;};  //Destructor
  tensor(int trank, ...);  //Constrctor taking rank, and size along each dimension.
  tensor(int trank, int *tdim); //Constructor taking rank, and array of sizes along the dimensions
  double get_elem(int ind, ...);  //get element at set of indeces, separated by commas
  double get_elem(int *inds);  //get element at indeces contained in array *inds
  void set_elem(double new_elem, ...);   //set element to new_elem at indeces, separated by commas
  void set_elem(double new_elem, int *inds);  //set elem to new_elem at indeces specified by array *inds
  tensor apply_sym(sym_op op);            // applies op.csym_mat^T* K * op.csym_mat
  void print(ostream &stream);

};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class specie{
 public:
  //char name[2];  // commented by jishnu
  string name; // jishnu
  int spin;
  double mass;
  double magmom;  // jishnu
  double U;	  // jishnu  // this is for LDA+U calculations
  double J;	  // jishnu  // this is for LDA+U calculations

  specie(){name =""; spin =0; mass =0.0; magmom =0.0; U =0.0;J = 0.0;}
  void print(ostream &stream);
};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class atompos{
 public:
  //Cluster expansion related variables
  int bit;
  specie occ;
  vector<specie> compon;
  double fcoord[3],ccoord[3];     //atom coordinates
  double dfcoord[3],dccoord[3];   //difference from ideal atom position
  double delta;
  ////////////////////////////////////////////////////////////////////////////////
  //added for occupation basis by Ben Swoboda
  vector<int> p_vec;              //occupation basis vector
  vector<int> spin_vec;           //vector of spins to be used as basis_vec if flagged
  vector<int> basis_vec;          //vector that stores the values of the basis in use
  char basis_flag;                //charachter 0,1,2,etc. that indicates which basis to use
                                  //nothing or 0=spin-basis, 1=occ-basis
  ////////////////////////////////////////////////////////////////////////////////

  //Cluster expansion related functions
  atompos();
  atompos apply_sym(sym_op op);
  void get_cart(double FtoC[3][3]);
  void get_frac(double CtoF[3][3]);
  void readf(istream &stream);
  void readc(istream &stream);
  void print(ostream &stream);
  void assign_spin();              //the first compon[0] has the highest spin
  int get_spin(string name);      //returns the spin of name[2]

  //Monte Carlo related variables
  int shift[4];                    //gives coordinates of unit cell and basis relative to prim
  vector< vector<int> > flip;      //for each compon, this gives the spins of the other components
  vector<double> mu;               //contains the chemical potentials for each specie with i<compon.size()-1, while mu[compon.size()-1]=0

  //Monte Carlo related functions
  void assemble_flip();
  int iflip(int spin);             //given a spin this gives the index in flip and dmu for that specie

};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class cluster{
 public:
  double min_leng,max_leng;
  vector<atompos> point;
  vector<sym_op> clust_group;
  //double clustmat[3][3];   // could be a force constant matrix, or a measure of strain for that cluster
  // we want to generalize this to a tensor object

  cluster(){min_leng=0; max_leng=0;}
  cluster apply_sym(sym_op op);
  void get_dimensions();
  void get_cart(double FtoC[3][3]);
  void get_frac(double CtoF[3][3]);
  void readf(istream &stream, int np);
  void readc(istream &stream, int np);
  void print(ostream &stream);
  void write_clust_group(ostream &stream);
  void determine_site_attributes(structure prim);
};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class orbit{
 public:
  vector<cluster> equiv;
  double eci;
  int stability;  // added by jishnu to determine environment stability in monte carlo
  //vector<sym_op> orb_group;  // this will be a set of matrices that link all equiv to the first one


  orbit(){eci=0.0;stability=0;};
  void get_cart(double FtoC[3][3]);
  void get_frac(double CtoF[3][3]);
  void readf(istream &stream, int np, int mult);
  void readc(istream &stream, int np, int mult);
  void print(ostream &stream);
  void determine_site_attributes(structure prim);
};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class multiplet{
 public:
  vector< vector<orbit> > orb;

  vector<int> size;
  vector<int> order;
  vector< vector<int> > index;
  vector< vector<int> > subcluster;

  void readf(istream &stream);
  void readc(istream &stream);
  void print(ostream &stream);
  void get_cart(double FtoC[3][3]);
  void get_frac(double CtoF[3][3]);
  void sort(int np);
  void get_index();
  void get_hierarchy();
  void print_hierarchy(ostream &stream);
  void read_eci(istream &stream);
  void determine_site_attributes(structure prim);
};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class structure{
 public:

  //lattice related variables

  char title[200];
  double scale;
  double lat[3][3];                   // cartesian coordinates of the lattice vectors (rows)
  double ilat[3][3];                  // ideal cartesian coordinates of the lattice vectors (rows) - either unrelaxed, or slat[][]*prim.lat[][]
  double slat[3][3];                  // supercell coordinates in terms of a primitive lattice (could be identity matrix)
  double strain[3][3];                //
  double latparam[3],latangle[3];
  int permut[3];
  double ilatparam[3],ilatangle[3];
  int ipermut[3];
  double FtoC[3][3],CtoF[3][3];       // Fractional to Cartesian trans mat and vice versa
  double PtoS[3][3],StoP[3][3];       // Primitive to Supercell trans mat and vice versa
  vector<sym_op> point_group;
  vector<vec> prim_grid;

  //basis related variables

  bool frac_on,cart_on;
  vector<int> num_each_specie;
  vector<specie> compon;
  vector<atompos> atom;
  vector<sym_op> factor_group;

  //reciprocal lattice variables

  double recip_lat[3][3];             // cartesian coordinates of the reciprocal lattice
  double recip_latparam[3],recip_latangle[3];
  int recip_permut[3];



  structure();

  //lattice related routines

  void get_trans_mat(){ coord_trans_mat(lat,FtoC,CtoF); coord_trans_mat(slat,StoP,PtoS);}
  void get_latparam();
  void get_ideal_latparam();
  void calc_point_group();
  void update_lat();
  void generate_3d_supercells(vector<structure> &supercell, int max_vol);
  void generate_2d_supercells(vector<structure> &supercell, int max_vol, int excluded_axis);
  void generate_3d_reduced_cell();
  void generate_2d_reduced_cell(int excluded_axis);
  void generate_slat(structure prim);                     //generates slat from lat
  void generate_lat(structure prim);                     //generates slat from lat   // added by jishnu
  void generate_slat(structure prim, double rescale);     //generates slat from lat after rescaling with rescale
  void generate_ideal_slat(structure prim);             //generates slat from ilat
  void generate_ideal_slat(structure prim, double rescale);
  void calc_strain();
  void generate_prim_grid();

  void read_lat_poscar(istream &stream);
  void write_lat_poscar(ostream &stream);
  void write_point_group();


  //lattice+basis related routines

  void calc_fractional();
  void calc_cartesian();
  void bring_in_cell();
  void calc_factor_group();
  void expand_prim_basis(structure prim);
  void map_on_expanded_prim_basis(structure prim);
  void map_on_expanded_prim_basis(structure prim, arrangement &conf);
  void idealize();
  void expand_prim_clust(multiplet basiplet, multiplet &super_basiplet);
  void collect_components();
  void collect_relax(string dir_name);      // reads in POS and CONTCAR and fills lat, rlat, atom etc.
  void update_struc();


  ////////////////////////////////////////////////////////////////////////////////
  //added by Ben Swoboda
  void read_struc_prim(istream &stream);
  ////////////////////////////////////////////////////////////////////////////////
  void read_species();     // added by jishnu
  void read_struc_poscar(istream &stream);
  void write_struc_poscar(ostream &stream);
  void write_struc_xyz(ostream &stream);
  void write_struc_xyz(ostream &stream, concentration out_conc);
  void write_factor_group();

  //reciprocal lattice related routines

  void calc_recip_lat();
  void get_recip_latparam();

};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class concentration{
 public:
  vector< vector<specie> > compon;
  vector< vector<double> > occup;
  vector< vector<double> > mu;

  void collect_components(structure &prim);
  void calc_concentration(structure &struc);
  void print_concentration(ostream &stream);
  void print_concentration_without_names(ostream &stream);
  void print_names(ostream &stream);
  void get_occup(istream &stream);  // added by jishnu

  //Monte Carlo related functions
  void set_zero();
  void increment(concentration conc);
  void normalize(int n);

};




//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class arrangement{
 public:
  int ns,nc;                       // supercell index and configuration index
  vector<int> bit;
  concentration conc;
  vector<double> correlations;
  string name;
  double energy,fenergy,cefenergy,fpfenergy;  // first principles energy, formation energy, cluster expanded formation energy // added by jishnu
  int fp,ce,te; // if fp=0, fenergy != fpfenergy and fp=1, fenergy = fpfenergy and same is true for ce and te // te stands for total energy  // added by jishnu
  double delE_from_facet; // added by jishnu // this is energy differnce from the hull
  //double norm_dist_from_facet;  // added by jishnu // this is normal distance from the hull
  double weight,reduction;
  vector<double> coordinate;   // contains concentration and energy of the arrangement
  // vector<double> coordinate_CE;  //contains concentration and CE_enenrgy of the arrangement // added by jishnu
  bool calculated,make,got_fenergy,got_cefenergy;   // got_fenergy is true when formation energy calculation is done for that arranegment // added by jishnu
  int relax_step;  //added by jishnu // to get the no of relaxation steps in the final vasp calculation

  arrangement(){calculated = false; make = false; got_fenergy = false; got_cefenergy =false; weight = 0.0; fp =0; ce=0; te =0; reduction=0;}
  void assemble_coordinate_fenergy();          //assemble coordinate vector using first-principles formation energy // added by jishnu
  // void assemble_coordinate_CE();          //assemble coordinate vector using Cluster Expansion formation energy // added by jishnu
  void print_bit(ostream &stream);
  void print_correlations(ostream &stream);
  void print_coordinate(ostream &stream);
  void print_coordinate_ternary(ostream &stream);  // added by jishnu
  void get_bit(istream &stream);  // added by jishnu
  void update_ce(); // to be made to convert fenergy = cefenergy and update coordinate, flag to indicate what is in fenergy // added by jishnu
  void update_fp(); // same but fenergy=fpfenergy   // added by jishnu
  void update_te(); // same but fenergy = energy   // added by jishnu
  //need routine that copy fenergy into fpfenergy or cefenergy
  void print_in_energy_file(ostream &stream); // added by jishnu
};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class superstructure{
 public:
  structure struc,curr_struc;
  vector<arrangement> conf;
  double kmesh[3];
  int nodes,ppn,walltime;  // added by jishnu
  string queue,parent_directory;  // added by jishnu
  vector< vector< vector< vector< int > > > > corr_to_atom_vec;  //added by John - associates basis functions with various curr_struc.atom[i]
  vector< vector< vector< int > > > basis_to_bit_vec;  //added by John - associates various curr_struc.atom[i].bit with appropriate spin/occupation basis values


  void decorate_superstructure(arrangement conf);
  void determine_kpoint_grid(double kpoint_dens);
  void print(string dir_name, string file_name);
  void print_incar(string dir_name);    // jishnu
  void print_potcar(string dir_name);
  void print_kpoint(string dir_name);
  void print_yihaw(string dir_name);   // added by jishnu // do not use this routine
  void read_yihaw_input();  // added by jishnu // do not use this one
};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class facet{
 public:
  vector<arrangement> corner;
  vector < double > normal_vec; // added by jishnu
  double offset; // added by jishnu
  vector<double> mu;

  void get_norm_vec(); // finds the coefficients of the facet plane equation ax+by+cz+d = 0; // added by jishnu
  bool find_endiff(arrangement arr, double &delE_from_facet); // sees whether that facet contains an arrangement and if yes, finds out the energy on facet // added by jishnu
  void get_mu(); // added by jishnu
};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class hull{
 public:
  vector<arrangement> point;
  //vector<structure> struc;
  vector<facet> face;
  //svector< vector<int> > point_to_face;

  void sort_conc();
  // bool below_hull(arrangement conf);

  //void assemble_coordinate_fenergy();
  void write_hull();  // added by jishnu
  void write_clex_hull();  // added by jishnu
  void clear_arrays();  // added by jishnu

  //routine to calculate the cluster expanded energy for each hull point
  //routines that update all coordinates with either the FP energy or the CE energy

  //for each point on the hull keep track of all facets that contain that point
  //the chemical potentials that stabilize the facets are the chemical potential bounds for the point
  //have something to make a chemical potential stability map (phase diagram)
};


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class configurations{
 public:
  structure prim;
  multiplet basiplet;
  vector<superstructure> superstruc;
  vector<arrangement> reference;       // contains the reference states to calculate formation energies
  hull chull;

  void generate_configurations(vector<structure> supercells);
  void generate_configurations_fast(vector<structure> supercells); //Added by John
  void generate_vasp_input_directories();
  void print_con();
  void print_con_old();    // added by jishnu
  void read_con();    // added by jishnu
  void read_corr();    // added by jishnu
  void print_corr();
  void read_energy_and_corr(); //added by jishnu
  void print_corr_old();    // added by jishnu
  void print_make_dirs();    // added by jishnu
  void read_make_dirs();    // added by jishnu
  void collect_reference();
  void collect_energies();
  void collect_energies_fast(); //Added by John
  void calculate_formation_energy();
  // void find_CE_formation_energies();  // jishnu
  // void get_CE_hull(); // added by jishnu // this is not the actual hull but the recalculation of FP-hull with CE.
  // void write_below_hull(); // added by jishnu // find distance from CE-hull,write it in below hull,label them as fitting/calculated/non-fitting but calculated etc
  //	// also do the mapbelowhull part in write_below_hull
  void assemble_hull();  // modified and rewritten by jishnu
  void CEfenergy_analysis();  // added by jishnu
  void get_delE_from_hull();  // added by jishnu // you can use this to weight according to distance from hull // right now not being used
  void get_delE_from_hull_w_clexen();  // added by jishnu
  void print_eci_inputfiles();
  void print_eci_inputfiles_old();  // added by jishnu
  void assemble_coordinate_fenergy();
  void cluster_expanded_energy();
  void reconstruct_from_read_files();   //added by jishnu
};


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class chempot{
 public:
  vector< vector<double> > m;
  vector< vector<specie> > compon;

  void initialize(concentration conc);      //sets up the vector structure of mu that is compatible with the concentration
  void initialize(vector<vector< specie > > init_compon);
  void set(facet face);                     //set the values of mu to be equal to those that stabilize the given facet
  void increment(chempot dmu);
  void print(ostream &stream);
  void print_compon(ostream &stream);

};



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class trajectory{
 public:
  vector< vector<double> > Rx;
  vector< vector<double> > Ry;
  vector< vector<double> > Rz;
  vector< vector<double> > R2;
  vector< vector<specie> > elements;
  vector< vector<int> > spin;


  void initialize(concentration conc);      //sets up the vector structure of mu that is compatible with the concentration
  void set_zero();
  void increment(trajectory R);
  void normalize(double D);
  void normalize(concentration conc);
  void print(ostream &stream);
  void print_elements(ostream &stream);
};


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class fluctuation{
 public:
  vector< vector<double> > f;
  vector< vector<specie> > compon;


  void initialize(concentration conc);
  void set_zero();
  void evaluate(concentration conc);
  void evaluate(trajectory R);
  void increment(fluctuation FF);
  void decrement(fluctuation FF);
  void normalize(double n);
  void normalize(double n, concentration conc);
  void print(ostream &stream);
  void print_elements(ostream &stream);
};


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//hop class collects possible hops for a particular basis site

class hop{
 public:
  int b;                                   // index of the basis
  int vac_spin_init;                       // spin of the vacancy on the initial site
  vector<int> vac_spin;                    // spin of the vacancy on the final site
  atompos initial;                         // initial site of the hop
  vector<cluster> endpoints;               // collects all clusters of possible hops (e.g. all equivalent nearest neighbors)
  vector<vec> jump_vec;
  vector<double> jump_leng;
  vector< mc_index > final;                // shifts of the final states of the hops
  vector< mc_index > activated;            // shifts of the activated states of the hop cluster
  vector< vector< mc_index > > reach;      // list of update sites for each endpoints cluster

  void get_reach(vector<multiplet> montiplet, bool clear, vector<atompos> basis);  // takes the montiplet and determines the reach for each endpoints cluster
  void print_hop_info(ostream &stream);

  // need to allow for repeated application of get_reach in case we have several cluster expansions simultaneously
  // have a boolean flag that clears the reach list, or enlarges it
};


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class mc_index{
 public:
  string name;
  int l;
  int shift[4];
  ////////////////////////////////////////////////////////////////////////////////
  //swoboda
  int num_specie;
  char basis_flag;
  ////////////////////////////////////////////////////////////////////////////////

  void print(ostream &stream);
  void print_name(ostream &stream);
  void print_shift(ostream &stream);
  void create_name(int i);
};




//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Monte_Carlo{
 public:
  // xph: KMC rateTable
  vector<vector< double > > rateTable;
  // how a movement(hop) tranfer a site to anohter 
  vector<vector<vector< int > > > movemap; // for basis[b] and  moves[movei],
                              // movemap[b][movei][0] is the connected basis
                              // movemap[b][movei][1:4] is the trans 
  // sites that interact with the center site.
  vector<vector<mc_index > > interaction_list;
  // sites that will move into the above interact sites.
  vector<vector<mc_index > > partial_update_list;
  int Lispin, Nispin, Vacspin; // save spins.

  // local_environment-energy maps for different basises.
  //vector<vector<pair<bitset<224>, double> > > energy_map;
  //vector<vector<pair<array<uint64_t, 4>, double> > > energy_map;
  vector<vector<pair<array<uint64_t, 5>, double> > > energy_map;
  //int ht_size; // the hash table size is 2^ht_size.
  //long htcap; // the hash table cap.
  vector<int> ht_size; // the hash table size is 2^ht_size.
  vector<uint32_t> htcap; // the hash table cap.
  vector<int> htload;
  vector<int> htncol;

  // routine to get the interaction_list 
  void get_interaction_list();
  // routine to get the partial update list 
  void get_partial_update();
  // routine to update rateTable[l]
  void update_rate_table(int l, double beta, bitset<nmoves> movetags);

  // routines for operations of Binary Index Tree (Fenwick Tree)
  // prefix sum, both query and update are in O(Logn) time.
  void updateBIT(double BITree[], int n, int index, double val);
  double getSum(double BITree[], int index);

  // routine to look up the local_environment-energy map first before calling the pointenergy()
  double get_pointenergy(int i, int j, int k, int b);

  vector<atompos> basis;
  concentration conc,num_atoms,sublat_conc,num_hops;   // work concentration variable that has the right dimensions
  vector<int> basis_to_conc;            // links each basis site to a concentration unit
  vector< vector<int> > conc_to_basis;  // links each concentration unit to all of its basis sites

  vector<int> basis_to_sublat;
  vector< vector<int> > sublat_to_basis;

  structure prim;
  structure Monte_Carlo_cell;
  multiplet basiplet;
  vector<multiplet> montiplet;
  vector< vector <hop> > jumps;    // for each basis site, there is a vector of hop objects
  vector<double> AVcorr;		   // Vector containing average correlations
  int di,dj,dk,db,ind1,ind2,ind3;
  int nuc;                         // number of unit cells
  int si,sj,sk;                    // shift variables used in index to insure a positive modulo
  int corr_flag;                   // Flag to indicate whether to calculate average correlations
  int *mcL;                        // unrolled Monte Carlo cell in a linear array
  int *ltoi,*ltoj,*ltok,*ltob;     // given the index of a site in the unrolled Monte Carlo cell mcL, it gives back i,j,k, or b
  double *Rx,*Ry,*Rz;                 // the trajectory of atom at site l - these have same length as mcL

  int *s;                          // unrolled shift factors, sequentially for each basis site, in a linear array
  int *nums;                       // number of sites for each cluster
  int *nump;                       // number of products for each eci
  int *mult;                       // multiplicity for each eci
  int *startend;                   // start and end indices for each basis
  double *eci;                     // eci vector
  double eci_empty;
  double *prob;                    //contains the probabilities for all hop events
  double *cprob;                   //contains the cumulative hop probabilities
  double tot_prob;
  int *ptol;                    // p is the index of prob and l is the index of mcL
  int *ptoj;                    // p is the index of prob and j is the index of a jump type for site l in ptol
  int *ptoh;                    // p is the index of prob and h is the index of the hop for site l in ptol
  int *ltosp;                   // l is the index of mcL sp=start p index for hops of site l
  int *arrayi, *arrayj, *arrayk;

  int nmcL,neci,ns,nse,np;           // need explicit sizes for all these arrays
  int idum;                        // random number integer used as seed

  concentration AVconc,AVnum_atoms,AVsublat_conc;
  fluctuation AVSusc,Susc;
  double AVenergy,heatcap,flipfreq;

  fluctuation AVkinL,kinL;
  trajectory Dtrace,corrfac;
  trajectory AVDtrace,AVcorrfac;
  trajectory R;
  double hop_leng;

  //Thermfac = inverse of Susc matrix (for each distinct sublattice) (inverse taken after multiplying Susc by kT)

  Monte_Carlo(structure in_prim, structure in_struc, multiplet in_basiplet, int idim, int jdim, int kdim);
  ~Monte_Carlo(){delete [] mcL; delete [] eci; delete [] s; delete [] nums; delete [] nump; delete [] mult; delete [] startend;
    delete [] arrayi; delete [] arrayj; delete[] arrayk;
  };

  void collect_basis();
  void collect_sublat();                 //maps basis sites onto crystallographically distinct sublattice sites
  void assemble_conc_basis_links();      //assembles the basis_to_conc and conc_to_basis vectors
  void update_mu(chempot mu);


  inline int index(int i, int j, int k, int b){
    return (b)*ind1+mdi(i,di)*ind2+mdj(j,dj)*ind3+mdk(k,dk);
  };
  inline int mdi(int i, int di) {return arrayi[i>=0 ? i : i+di];};  // -di <= i <= 2di - 1, e.g. di = 12, i is [-12, 23]
  inline int mdj(int j, int dj) {return arrayj[j>=0 ? j : j+dj];};
  inline int mdk(int k, int dk) {return arrayk[k>=0 ? k : k+dk];};
  inline void invert_index(int &i, int &j, int &k, int &b, int l);
  void generate_eci_arrays();
  void write_point_energy(ofstream &out);
  void write_point_corr(ofstream &out); //Added by John
  void write_normalized_point_energy(ofstream &out);
  void write_monte_h(string class_file);
  void write_monte_xyz(ostream &stream);


  double pointenergy(int i, int j, int k, int b);
  void  pointcorr(int i, int j, int k, int b);
  double normalized_pointenergy(int i, int j, int k, int b);
  void calc_energy(double &energy);
  void calc_concentration();
  void calc_num_atoms();
  void calc_sublat_concentration();
  void update_num_hops(int l, int ll, int b, int bb);
  double calc_grand_canonical_energy(chempot mu);




  //conventional Monte Carlo routines
  bool compatible(structure init_struc, int &ndi, int &ndj, int &ndk);
  void initialize(structure init_struc);
  void initialize(concentration conc);
  void initialize_1_specie(double conc);
  ////////////////////////////////////////////////////////////////////////////////
  //added by Ben Swoboda
  void initialize_1vac(concentration conc);  //1 vacancy in monte cell
  ////////////////////////////////////////////////////////////////////////////////
  void grand_canonical(double beta, chempot mu, int n_pass, int n_equil_pass);
  void canonical_single_species(double beta, int n_pass, int n_equil_pass, int n_writeout); //Added by Aziz
  void anneal_single_species(double T_init, double T_final, int n_pass); // added by AU
  double lte(double beta, chempot mu);
  //  void canonical(double beta, double n_pass, double n_equil_pass);
  //  void n_fold_grand_canonical(double beta, chempot mu, double n_pass, double n_equil_pass);


  //kinetic Monte Carlo routines
  void initialize_kmc();
  void extend_reach();
  void get_hop_prob(int i, int j, int k, int b, double beta);
  double calc_barrier(int i, int j, int k, int b, int ii, int jj, int kk, int bb, int l, int ll, int ht, int h);
  void initialize_prob(double beta);
  int pick_hop();
  void update_prob(int i, int j, int k, int b, int ht, int h, double beta);
  void kinetic(double beta, double n_pass, double n_equil_pass);

  void collect_R();

  void output_Monte_Carlo_cell();

};




//********************************************************************
//Routines

double determinant(double mat[3][3]);
void inverse(double mat[3][3], double invmat[3][3]);
void matrix_mult(double mat1[3][3], double mat2[3][3], double mat3[3][3]);
void get_perp(double vec1[3], double vec2[3]); //Added by John
void get_perp(double vec1[3], double vec2[3], double vec3[3]); //Added by John
bool normalize(double vec1[3], double length); //Added by John -- returns false if vector is null
void lat_dimension(double lat[3][3], double radius, int dim[3]);
bool compare(double mat1[3][3], double mat2[3][3]);
bool compare(double vec1[3], double vec2[3]);
bool compare(double vec1[3], double vec2[3], int trans[3]);
bool compare(vector<double> vec1, vector<double> vec2);
bool compare(char name1[2], char name2[2]);
bool compare(specie compon1, specie compon2);
bool compare(vector<specie> compon1, vector<specie> compon2);
bool compare(atompos &atom1, atompos &atom2);
bool compare(atompos atom1, atompos atom2, int trans[3]);
bool compare(cluster &clust1, cluster &clust2);
////////////////////////////////////////////////////////////////////////////////
//added by anton
bool compare(orbit orb1, orbit orb2);
////////////////////////////////////////////////////////////////////////////////
bool compare(concentration conc1, concentration conc2);
bool compare(mc_index m1, mc_index m2);
bool new_mc_index(vector<mc_index> v1, mc_index m2);
bool is_integer(double vec[3]);
bool is_integer(double mat[3][3]);
void within(double fcoord[3]);
void within(atompos &atom);
void within(cluster &clust);
void within(cluster &clust, int n);
////////////////////////////////////////////////////////////////////////////////
//added by Ben Swoboda
void within(structure &struc);
////////////////////////////////////////////////////////////////////////////////
void latticeparam(double lat[3][3], double latparam[3], double latangle[3]);
void latticeparam(double lat[3][3], double latparam[3], double latangle[3], int permut[3]);
void conv_AtoB(double AtoB[3][3], double Acoord[3], double Bcoord[3]);
double distance(atompos atom1, atompos atom2);
bool update_bit(vector<int> max_bit, vector<int> &bit, int &last);
void get_equiv(orbit &orb, vector<sym_op> &op);
bool new_clust(cluster clust, orbit &orb);
bool new_clust(cluster clust, vector<orbit> &orbvec);
void get_loc_equiv(orbit &orb, vector<sym_op> &op);
bool new_loc_clust(cluster clust, orbit orb);
bool new_loc_clust(cluster clust, vector<orbit> torbvec);
void calc_correlations(structure struc, multiplet super_basiplet, arrangement &conf);
void get_super_basis_vec(structure &superstruc, vector < vector < vector < int > > > &super_basis_vec); //Added by John
void get_corr_vector(structure &struc, multiplet &super_basiplet, vector < vector < vector < vector < int > > > > &corr_vector); //Added by John
bool new_conf(arrangement &conf,superstructure &superstruc);
bool new_conf(arrangement &conf,vector<superstructure> &superstruc);
void get_shift(atompos &atom, vector<atompos> basis);
void double_to_string(double n, string &a, int dec_places); //Added by John
void int_to_string(int i, string &a, int base);
void generate_ext_clust(structure struc, int min_num_compon, int max_num_points,vector<double> max_radius, multiplet &clustiplet);
void generate_ext_basis(structure struc, multiplet clustiplet, multiplet &basiplet);
void generate_ext_monteclust(vector<atompos> basis, multiplet basiplet, vector<multiplet> &montiplet);
////////////////////////////////////////////////////////////////////////////////
//added by anton - filters a multiplet for clusters containing just one activated site (with occupation basis = 1)
void filter_activated_clust(multiplet clustiplet);
void merge_multiplets(multiplet clustiplet1, multiplet clustiplet2, multiplet &clustiplet3);
void write_clust(multiplet clustiplet, string out_file);
void write_fclust(multiplet clustiplet, string out_file);
////////////////////////////////////////////////////////////////////////////////

bool scandirectory(string dirname, string filename);
bool read_oszicar(string dirname, double& e0);
bool read_oszicar(string dirname, double& e0, int &count);   // added by jishnu
bool read_mc_input(string cond_file, int &n_pass, int &n_equil_pass, int &nx, int &ny, int &nz, chempot &muinit, chempot &mu_min, chempot &mu_max, vector<chempot> &muinc, double $Tinit, double &Tmin, double &Tmax, double &Tinc, int &xyz_step, int &corr_flag, int &temp_chem);
double ran0(int &idum);

////////////////////////////////////////////////////////////////////////////////
//added by Ben Swoboda to utilize multiple basis
void get_clust_func(atompos atom1, atompos atom2, double &clust_func);
void get_basis_vectors(atompos &atom);
////////////////////////////////////////////////////////////////////////////////
void read_junk(istream &stream);  // added by jishnu
//int hullfinder_bi(double); // added by jishnu // this generates binary hull and cutts off the high energy structres // made member functions of cofigurations
// int hullfinder_ter(double); // added by jishnu // this generates ternary hull and cutts off the high energy structres // made member functions of cofigurations

//************************************************************

double vec::calc_dist(){
  if(cart_on==false){
    cout << "no cartesian coordinates for vec \n";
    exit(1);
  }

  length=0;
  for(int i=0; i<3; i++)
    length=length+ccoord[i]*ccoord[i];
  length=sqrt(length);

  return length;
}


//************************************************************

vec vec::apply_sym(sym_op op){
  int i,j,k;
  vec tlat;

  if(op.frac_on == false || op.cart_on == false)op.update();


  for(i=0; i<3; i++){
    tlat.fcoord[i]=op.ftau[i];
    tlat.ccoord[i]=op.ctau[i];
    for(j=0; j<3; j++){
      tlat.fcoord[i]=tlat.fcoord[i]+op.fsym_mat[i][j]*fcoord[j];
      tlat.ccoord[i]=tlat.ccoord[i]+op.csym_mat[i][j]*ccoord[j];
    }
  }
  return tlat;
}


//************************************************************

void vec::print_frac(ostream &stream){

  for(int i=0; i<3; i++){
    stream.precision(5);stream.width(10);stream.setf(ios::showpoint);
    stream << fcoord[i] << " ";
  }
  stream << "\n";
}

//************************************************************

void vec::print_cart(ostream &stream){

  for(int i=0; i<3; i++){
    stream.precision(5);stream.width(10);stream.setf(ios::showpoint);
    stream << ccoord[i] << " ";
  }
  stream << "\n";
}

//************************************************************
//Edited by John
tensor tensor::apply_sym(sym_op op){
  // calculates the op.csym_mat^T * K * op.csym_mat

  tensor ttensor(rank, dim);
  bool D_flag=true;
  for(int i=0; i<rank; i++) D_flag=D_flag&&(dim[i]==3);
  if(!D_flag){
    cout << "ERROR:  Attempting apply 3D transformation to tensor with improper dimensionality.\n";
    return ttensor;
  }
  int *elem_array=new int[rank];
  int *sub_array=new int[rank];
  bool elem_cont=true;
  for(int i=0; i<rank; i++)
    elem_array[i]=0;
  while(elem_cont){
    double telem=0.0;
    for(int i=0; i<rank; i++)
      sub_array[i]=0;
    bool sub_cont=true;
    while(sub_cont){
      double ttelem=1.0;
      for(int i=0; i<rank; i++){
        ttelem*=op.csym_mat[elem_array[i]][sub_array[i]];
      }
      ttelem*=get_elem(sub_array);
      telem+=ttelem;
      for(int i=0; i<rank; i++){
	sub_array[i]+=1;
	if(sub_array[i]==dim[i])
          sub_array[i]=0;
	else break;
        if(i==rank-1) sub_cont=false;
      }
    }
    ttensor.set_elem(telem, elem_array);

    for(int i=0; i<rank; i++){
      elem_array[i]=elem_array[i]+1;
      if(elem_array[i]>=dim[i])
        elem_array[i]=0;
      else break;
      if(i==rank-1) elem_cont=false;
    }
  }
  return ttensor;
}
//\Edited by John

//************************************************************

//Added by John
//Copy constructor
tensor::tensor(tensor const& ttens){
  rank=ttens.rank;
  size=ttens.size;
  dim=new int[rank];
  mult=new int[rank];
  K=new double[size];
  for(int i=0; i<rank; i++){
    dim[i]=ttens.dim[i];
    mult[i]=ttens.mult[i];
  }
  for(int j=0; j<size; j++)
    K[j]=ttens.K[j];
}

//************************************************************

//Added by John
//Default constructor
tensor::tensor(){
  dim=0;
  K=0;
  mult=0;
}

//************************************************************

//Added by John
tensor::tensor(int trank, ...){
  size=1;
  rank=trank;
  va_list argPtr;
  va_start( argPtr, trank );
  dim=new int[rank];
  mult=new int[rank];
  for(int i=0; i<rank; i++){
    dim[i]=va_arg( argPtr, int );
    mult[i]=size;
    size*=dim[i];
  }
  va_end(argPtr);
  K=new double[size];
  for(int i=0; i<size; i++)
    K[i]=0.0;
}

//\Added by John

//************************************************************

//Added by John
tensor::tensor(int trank, int *tdim){
  size=1;
  rank=trank;
  dim=new int[rank];
  mult=new int[rank];
  for(int i=0; i<rank; i++){
    dim[i]=tdim[i];
    mult[i]=size;
    size*=dim[i];
  }
  K=new double[size];
  for(int i=0; i<size; i++)
    K[i]=0.0;
}

//\Added by John

//************************************************************

//Added by John
//Assignment operator
tensor& tensor::operator=(const tensor& ttens){
  rank = ttens.rank;
  size = ttens.size;
  delete dim;
  delete mult;
  delete K;
  dim = new int[rank];
  mult = new int[rank];
  K = new double[size];
  for(int i=0; i<rank; i++){
    dim[i]=ttens.dim[i];
    mult[i]=ttens.mult[i];
  }
  for(int i=0; i<size; i++)
    K[i]=ttens.K[i];

  return *this;
}

//************************************************************

//Added by John
  double tensor::get_elem(int ind, ...){
    va_list argPtr;
    va_start(argPtr, ind);
    if(!(ind>-1 && ind<dim[0])){
      cout << "WARNING:  Attempted to acess tensor element out of bounds.";
      return 0.0;
    }
    int ctr=ind*mult[0];

    for(int i=1; i<rank; i++){
      int tind=va_arg( argPtr, int );
      if(tind>-1 && tind <dim[i])
	ctr+=tind*mult[i];
      else{
	cout << "WARNING:  Attempted to acess tensor element out of bounds.";
	return 0.0;
      }
    }
    va_end(argPtr);
    return K[ctr];
  }
//\Added by John

//************************************************************

//Added by John
double tensor::get_elem(int *inds){
  int ctr=0;
  for(int i=0; i<rank; i++){
    if(inds[i]>-1 && inds[i] <dim[i])
      ctr+=inds[i]*mult[i];
    else{
      cout << "WARNING:  Attempted to acess tensor element out of bounds.";
      return 0.0;
    }
  }
  return K[ctr];
}
//\Added by John

//************************************************************
//Added by John
void tensor::set_elem(double new_elem, ...){
  va_list argPtr;
  va_start(argPtr, new_elem);
  int ctr=0;
  for(int i=0; i<rank; i++){
    int tind=va_arg(argPtr, int);
    if(tind>-1 && tind <dim[i])
      ctr+=tind*mult[i];
    else{
      cout << "WARNING:  Attempted to acess tensor element out of bounds.";
      return;
    }
  }
  va_end(argPtr);
  K[ctr]=new_elem;
  return;
}
//\Added by John

//************************************************************
//Added by John
void tensor::set_elem(double new_elem, int *inds){
  int ctr=0;
  //  cout << "Inside set_elem, new_elem =" << new_elem << ";  rank =" << rank << "\n";
  for(int i=0; i<rank; i++){

    if(inds[i]>=0 && inds[i] <dim[i]){
      ctr=ctr+inds[i]*mult[i];
      //      cout << "ctr=" << ctr << " and inds[" << i << "]=" << inds[i] << "\n";
    }
    else{
      cout << "WARNING:  Attempted to acess tensor element out of bounds.";
      return;
    }
  }
  //  cout << "\n";
  K[ctr]=new_elem;
  return;
}
//\Added by John

//************************************************************

//Added by John
void tensor::print(ostream &stream){

  for(int i=0; i<size; i++){
    stream << "   " << K[i];
    for(int j=0; j<rank; j++){
      if(!((i+1)%mult[j])&&mult[j]!=1)
        stream << "\n";
    }
  }
}
//\Addition


//************************************************************
//Added by John
void sym_op::get_sym_type(){

  int i, j;
  double vec_sum, vec_mag;
  if(!cart_on){
    update();
  }

  double det=0.0;
  double trace=0.0;
  double tmat[3][3];

  //Copy csym_mat to temporary
  for(i=0; i<3; i++){
    for(j=0; j<3; j++){
      tmat[i][j]=csym_mat[i][j];
    }
  }
  //Get Determinant and trace
  det=determinant(tmat);

  for(i=0; i<3; i++){
    trace+=tmat[i][i];
  }


  if(abs(trace-3.0)<tol){ //Sym_op is identity
    sym_type=0;
    return;
  }

  if(abs(trace+3.0)<tol){ //Sym_op is inversion
    sym_type=-1;
    return;
  }


  if(det<0 && abs(trace-1.0)<tol){ //operation is mirror
    //The trace criterion can be shown by noting that a mirror
    //is a 180 degree rotation composed with an inversion

    sym_type=2;

    //Mirror planes have eigenvalues 1,1,-1; the Eigenvectors form an
    //orthonormal basis into which any member of R^3 can be decomposed.
    //For mirror operation S and test vector v, we take the vector
    //w=v-S*v to be the eigenvector with eigenvalue -1.  We must test as many
    //as 3 cases to ensure that test vector v is not coplanar with mirror
    double vec1[3], vec2[3];
    vec1[0]=1;  vec1[1]=1;  vec1[2]=1;
    conv_AtoB(tmat, vec1, vec2);
    vec_sum=0.0;
    for(i=0; i<3; i++){
      vec2[i]-=vec1[i];
      if(abs(vec_sum)<tol && abs(vec2[i])>tol)
	vec_sum=vec2[i]/abs(vec2[i]);
      eigenvec.ccoord[i]=vec2[i];
    }
    if(normalize(eigenvec.ccoord, vec_sum)){
      conv_AtoB(CtoF, eigenvec.ccoord, eigenvec.fcoord);
      eigenvec.frac_on=true;
      eigenvec.cart_on=true;
      return;
    }

    vec1[2]=-2;
    conv_AtoB(tmat, vec1, vec2);
    vec_sum=0.0;
    for(i=0; i<3; i++){
      vec2[i]-=vec1[i];
      if(abs(vec_sum)<tol && abs(vec2[i])>tol)
	vec_sum=vec2[i]/abs(vec2[i]);
      eigenvec.ccoord[i]=vec2[i];
    }
    if(normalize(eigenvec.ccoord, vec_sum)){
      conv_AtoB(CtoF, eigenvec.ccoord, eigenvec.fcoord);
      eigenvec.frac_on=true;
      eigenvec.cart_on=true;
      return;
    }

    eigenvec.ccoord[0]=1.0/sqrt(2);
    eigenvec.ccoord[1]=-1.0/sqrt(2);
    eigenvec.ccoord[2]=0.0;
    conv_AtoB(CtoF, eigenvec.ccoord, eigenvec.fcoord);
    eigenvec.frac_on=true;
    eigenvec.cart_on=true;
    return;
  }//\End Mirror Plane Conditions

  else { //operation is either rotation or rotoinversion
    if(det<(-tol)){ //operation is rotoinversion
      trace*=-1;
      det*=-1;
      sym_type=3;
      //compose rotoinversion with inversion so that angle and axis may be extracted
      //in same was as proper rotation
      for(i=0; i<3; i++){
	for(j=0; j<3; j++){
	  tmat[i][j]*=-1;
	}
      }
    }
    else sym_type=1; //operation is rotation

    if(abs(trace+1)<tol){ //rotation is 180 degrees, which requires special care, since rotation matrix becomes symmetric
      //180 rotation can be decomposed into two orthogonal mirror planes, so we use similar method as above, but finding +1 eigenvector
      double vec1[3], vec2[3];
      op_angle=180;
      vec1[0]=1;  vec1[1]=1;  vec1[2]=1;
      conv_AtoB(tmat, vec1, vec2);
      double vec_sum=0.0;
      for(i=0; i<3; i++){
	vec2[i]+=vec1[i];
	if(abs(vec_sum)<tol && abs(vec2[i])>tol)
	  vec_sum=vec2[i]/abs(vec2[i]);
	eigenvec.ccoord[i]=vec2[i];
      }
      if(normalize(eigenvec.ccoord, vec_sum)){
	conv_AtoB(CtoF, eigenvec.ccoord, eigenvec.fcoord);
	eigenvec.frac_on=true;
	eigenvec.cart_on=true;
	return;
      }

      vec1[2]=-2;
      conv_AtoB(tmat, vec1, vec2);
      vec_sum=0.0;
      for(i=0; i<3; i++){
	vec2[i]+=vec1[i];
	if(abs(vec_sum)<tol && abs(vec2[i])>tol)
	  vec_sum=vec2[i]/abs(vec2[i]);
	eigenvec.ccoord[i]=vec2[i];
      }
      if(normalize(eigenvec.ccoord, vec_sum)){
	conv_AtoB(CtoF, eigenvec.ccoord, eigenvec.fcoord);
	eigenvec.frac_on=true;
	eigenvec.cart_on=true;
	return;
      }

      eigenvec.ccoord[0]=1.0/sqrt(2);
      eigenvec.ccoord[1]=-1.0/sqrt(2);
      eigenvec.ccoord[2]=0.0;
      conv_AtoB(CtoF, eigenvec.ccoord, eigenvec.fcoord);
      eigenvec.frac_on=true;
      eigenvec.cart_on=true;
      return;
    }

    // Following only evaluates if we have non-180 proper rotation
    // Method uses inversion of axis-angle interpretation of a rotation matrix R
    // With axis v=(x,y,z) and angle ÅŒÅ∏, with ||v||=1
    //  c = cos(ÅŒÅ∏); s = sin(ÅŒÅ∏); C = 1-c
    //      [ x*xC+c   xyC-zs   zxC+ys ]
    //  R = [ xyC+zs   y*yC+c   yzC-xs ]
    //      [ zxC-ys   yzC+xs   z*zC+c ]
    double tangle;
    vec_sum=0.0;
    vec_mag=0.0;
    for(i=0; i<3; i++){
      eigenvec.ccoord[i]=tmat[(i+2)%3][(i+1)%3]-tmat[(i+1)%3][(i+2)%3];
      vec_mag += eigenvec.ccoord[i]*eigenvec.ccoord[i];
      if(abs(vec_sum)<tol && abs(eigenvec.ccoord[i])>tol)
	vec_sum=eigenvec.ccoord[i]/abs(eigenvec.ccoord[i]);
    }
    vec_mag=sqrt(vec_mag);
    tangle=round((180.0/3.141592654)*atan2(vec_mag,trace-1));
    op_angle=int(tangle);
    normalize(eigenvec.ccoord,vec_sum);
    if(vec_sum<0){
      op_angle=360-op_angle;
    }

    conv_AtoB(CtoF, eigenvec.ccoord, eigenvec.fcoord);
    eigenvec.frac_on=true;
    eigenvec.cart_on=true;
    return;
  }
}
//\End Addition

//************************************************************

void sym_op::print_fsym_mat(ostream &stream){
  //Added by John

  if(sym_type==-1) stream << "Inversion Operation: \n";
  if(!sym_type) stream << "Identity Operation: \n";
  if(sym_type==1){
    stream << op_angle << " degree Rotation (or screw) Operation about axis: ";
    eigenvec.print_frac(stream);
  }
  if(sym_type==2){
    stream << "Mirror (or glide) Operation with plane normal: ";
    eigenvec.print_frac(stream);
  }
  if(sym_type==3){
    stream << op_angle << " degree Rotoinversion (or screw) Operation about axis: ";
    eigenvec.print_frac(stream);
  }
  //\End Addition
  stream << "        symmetry operation matrix                  shift \n";
  for(int i=0; i<3; i++){
    stream.precision(9);stream.width(15);stream.setf(ios::showpoint);
    for(int j=0; j<3; j++) stream << fsym_mat[i][j] << "  ";
    stream << "       " << ftau[i] << "\n";
  }
}

//************************************************************

void sym_op::print_csym_mat(ostream &stream){
  //Added by John
  if(sym_type==-1) stream << "Inversion Operation: \n";
  if(!sym_type) stream << "Identity Operation: \n";
  if(sym_type==1){
    stream << op_angle << " degree Rotation (or screw) Operation about axis: ";
    eigenvec.print_cart(stream);
  }
  if(sym_type==2){
    stream << "Mirror (or glide) Operation with plane normal: ";
    eigenvec.print_cart(stream);
  }
  if(sym_type==3){
    stream << op_angle << " degree Rotoinversion (or screw) Operation about axis: ";
    eigenvec.print_cart(stream);
  }
  //\End Addition
  stream << "        symmetry operation matrix                  shift \n";
  for(int i=0; i<3; i++){
    stream.precision(9);stream.width(15);stream.setf(ios::showpoint);
    for(int j=0; j<3; j++) stream << csym_mat[i][j] << "  ";
    stream << "       " << ctau[i] << "\n";
  }
}


//************************************************************

void sym_op::get_csym_mat(){
  int i,j,k;
  double temp[3][3];


  if(cart_on == true) return;

  if(cart_on == false) {
    if(frac_on == false){
      cout << "No sym_op initialized - cannot get_csym_mat\n";
      return;
    }

    for(i=0; i<3; i++){
      for(j=0; j<3; j++){
	temp[i][j]=0;
	for(k=0; k<3; k++) temp[i][j]=temp[i][j]+fsym_mat[i][k]*CtoF[k][j];
      }
    }

    for(i=0; i<3; i++){
      for(j=0; j<3; j++){
	csym_mat[i][j]=0.0;
	for(k=0; k<3; k++) csym_mat[i][j]=csym_mat[i][j]+FtoC[i][k]*temp[k][j];
      }
    }

    for(i=0; i<3; i++){
      ctau[i]=0.0;
      for(j=0; j<3; j++) ctau[i]=ctau[i]+FtoC[i][j]*ftau[j];
    }
  }
  cart_on=true;
  return;
}



//************************************************************

void sym_op::get_fsym_mat(){
  int i,j,k;
  double temp[3][3];

  if(frac_on == true) return;
  if(frac_on == false) {
    if(cart_on == false){
      cout << "No sym_op initialized - cannot get_fsym_mat\n";
      return;
    }

    for(i=0; i<3; i++){
      for(j=0; j<3; j++){
	temp[i][j]=0;
	for(k=0; k<3; k++) temp[i][j]=temp[i][j]+csym_mat[i][k]*FtoC[k][j];
      }
    }

    for(i=0; i<3; i++){
      for(j=0; j<3; j++){
	fsym_mat[i][j]=0.0;
	for(k=0; k<3; k++) fsym_mat[i][j]=fsym_mat[i][j]+CtoF[i][k]*temp[k][j];
      }
    }

    for(i=0; i<3; i++){
      ftau[i]=0.0;
      for(j=0; j<3; j++) ftau[i]=ftau[i]+CtoF[i][j]*ctau[j];
    }
  }
  frac_on=true;
  return;
}


//************************************************************

void sym_op::update(){
  get_trans_mat();
  get_csym_mat();
  get_fsym_mat();
  return;
}


//************************************************************
structure::structure(){
  int i,j;
  for(i=0; i<200; i++) title[i]=0;
  for(i=0; i<3; i++){
    for(j=0; j<3; j++){
      lat[i][j]=0.0;
      slat[i][j]=0.0;
      FtoC[i][j]=0.0;
      CtoF[i][j]=0.0;
    }
    slat[i][i]=1.0;
  }

  frac_on=false;
  cart_on=false;

}


//************************************************************
void structure::get_latparam(){

  latticeparam(lat,latparam,latangle,permut);

  return;

}


//************************************************************
void structure::get_ideal_latparam(){

  latticeparam(ilat,ilatparam,ilatangle,ipermut);

  return;

}


//************************************************************
//read a POSCAR like file and collect all the structure variable

void structure::read_lat_poscar(istream &stream){
  int i,j;
  char ch;
  char buff[200];
  char sp[200];


  stream.getline(title,199);
  stream >> scale;
  for(i=0; i<3; i++){
    for(j=0; j<3; j++){
      stream >> lat[i][j];
    }
  }

  //normalize the scale to 1.0 and adjust lattice vectors accordingly

  for(i=0; i<3; i++)
    for(j=0; j<3; j++)lat[i][j]=scale*lat[i][j];
  scale=1.0;

  return;

}



//************************************************************
//write the structure to a file in POSCAR like format

void structure::write_lat_poscar(ostream &stream) {
  int i,j;

  stream << title <<"\n";

  stream.precision(7);stream.width(12);stream.setf(ios::showpoint);
  stream << scale <<"\n";

  for(int i=0; i<3; i++){
    stream << "  ";
    for(int j=0; j<3; j++){

      stream.precision(9);stream.width(15);stream.setf(ios::showpoint);
      stream << lat[i][j] << " ";

    }
    stream << "\n";
  }

  return;

}




//************************************************************

void structure::calc_point_group(){
  int i,j,k;
  int dim[3];
  double radius;
  vec temp;
  vector<vec> gridlat;
  double tlat[3][3],tlatparam[3],tlatangle[3];
  double tcsym_mat[3][3];
  sym_op tpoint_group;

  get_latparam();
  int num_point_group=0;

  //make a lattice parallelepiped that encompasses a sphere with radius = 2.1*largest latparam

  radius=1.5*latparam[permut[2]];

  lat_dimension(lat,radius,dim);

  for(i=0; i<3; i++){
    if(dim[i] > 3) dim[i]=3;
  }

  cout << "inside calc_point group \n";
  cout << "dimension = ";
  for(i=0; i<3; i++)cout << dim[i] << " ";
  cout << "\n";

  for(i=-dim[0]; i<=dim[0]; i++){
    for(j=-dim[1]; j<=dim[1]; j++){
      for(k=-dim[2]; k<=dim[2]; k++){
	if(!(i==0 && j==0 && k==0)){
	  temp.ccoord[0]=i*lat[0][0]+j*lat[1][0]+k*lat[2][0];
	  temp.ccoord[1]=i*lat[0][1]+j*lat[1][1]+k*lat[2][1];
	  temp.ccoord[2]=i*lat[0][2]+j*lat[1][2]+k*lat[2][2];
	  temp.cart_on=true;
	  temp.calc_dist();
	}

	//keep only the lattice points within the sphere with radius

	if(temp.length < radius){
	  gridlat.push_back(temp);
	}
      }
    }
  }


  cout << "made the grid \n";
  cout << "number of sites in the grid = " << gridlat.size() << "\n";

  //for each set of three lattice points within the sphere see which one has the
  //same sets of lengths and angles as the original lattice unit cell vectors.

  for(i=0; i<gridlat.size(); i++){
    for(j=0; j<gridlat.size(); j++){
      for(k=0; k<gridlat.size(); k++){

	if(i!=j && i!=k && j!=k){
	  for(int ii=0; ii<3; ii++){
	    tlat[0][ii]=gridlat[i].ccoord[ii];
	    tlat[1][ii]=gridlat[j].ccoord[ii];
	    tlat[2][ii]=gridlat[k].ccoord[ii];
	  }
	  latticeparam(tlat,tlatparam,tlatangle);

	  //compare the tlat... and lat... to see if they are the same lattice
	  // that is do the lattice vectors have the same lengths and the same angles

	  if(abs(latparam[0]-tlatparam[0]) < tol*latparam[permut[0]] &&
	     abs(latparam[1]-tlatparam[1]) < tol*latparam[permut[0]] &&
	     abs(latparam[2]-tlatparam[2]) < tol*latparam[permut[0]] &&
	     abs(latangle[0]-tlatangle[0]) < tol &&
	     abs(latangle[1]-tlatangle[1]) < tol &&
	     abs(latangle[2]-tlatangle[2]) < tol ){

	    // get the matrix that relates the two lattice vectors


	    for(int ii=0; ii<3; ii++){
	      for(int jj=0; jj<3; jj++){
		tcsym_mat[ii][jj]=0.0;
		for(int kk=0; kk<3; kk++)
		  tcsym_mat[ii][jj]=tcsym_mat[ii][jj]+tlat[kk][ii]*CtoF[kk][jj];
	      }
	    }

	    // check whether this symmetry operation is new or not

	    int ll=0;
	    for(int ii=0; ii<num_point_group; ii++)
	      if(compare(tcsym_mat,point_group[ii].csym_mat))break;
	      else ll++;

	    // if the symmetry operation is new, add it to the pointgroup array
	    // and update all info about the sym_op object

	    if(num_point_group == 0 || ll == num_point_group){
	      for(int jj=0; jj<3; jj++){
		tpoint_group.frac_on=false;
		tpoint_group.cart_on=false;
		tpoint_group.ctau[jj]=0.0;
		for(int kk=0; kk<3; kk++){
		  tpoint_group.csym_mat[jj][kk]=tcsym_mat[jj][kk];
		  tpoint_group.lat[jj][kk]=lat[jj][kk];
		}
	      }

	      tpoint_group.cart_on=true;
	      tpoint_group.update();
 	      tpoint_group.get_sym_type(); // Added by John
	      point_group.push_back(tpoint_group);
	      num_point_group++;
	    }


	  }
	}
      }
    }
  }
  cout << "finished finding all point group operartions \n";
}


//************************************************************
void structure::update_lat(){
  get_trans_mat();
  get_latparam();
  calc_point_group();
  return;
}


//************************************************************
void structure::write_point_group(){
  int pg;

  ofstream out("point_group");
  if(!out){
    cout << "Cannot open file.\n";
    return;
  }

  cout << " number of point group ops " << point_group.size() << "\n";

  for(pg=0; pg<point_group.size(); pg++){
    out << "point group operation " << pg << " \n";
    point_group[pg].print_fsym_mat(out);
    out << "\n";
  }

  out.close();
}


//************************************************************
void structure::generate_3d_supercells(vector<structure> &supercell, int max_vol){
  int vol,pg,i,j,k;
  int tslat[3][3];


  //algorithm relayed to me by Tim Mueller
  //make upper triangular matrix where the product of the diagonal elements equals the volume
  //then for the elements above the diagonal choose all values less than the diagonal element
  //for each lattice obtained this way, apply point group symmetry operations
  //see if the transformed superlattice can be written as a linear combination of superlattices already found
  //if not add it to the list


  for(vol = 1; vol <= max_vol; vol++){
    vector<structure> tsupercell;

    //generate all tslat[][] matrices that are upper diagonal where the product of the
    //diagonal equals the current volume vol and where the upper diagonal elements take
    //all values less than the diagonal below it

    //initialize the superlattice vectors to zero
    for(i=0; i<3; i++)
      for(j=0; j<3; j++)tslat[i][j]=0;

    for(tslat[0][0]=1; tslat[0][0]<=vol; tslat[0][0]++){
      if(vol%tslat[0][0] == 0){
	for(tslat[1][1]=1; tslat[1][1]<=vol/tslat[0][0]; tslat[1][1]++){
	  if((vol/tslat[0][0])%tslat[1][1] == 0){
	    tslat[2][2]=(vol/tslat[0][0])/tslat[1][1];
	    for(tslat[0][1]=0; tslat[0][1]<tslat[1][1]; tslat[0][1]++){
	      for(tslat[0][2]=0; tslat[0][2]<tslat[2][2]; tslat[0][2]++){
		for(tslat[1][2]=0; tslat[1][2]<tslat[2][2]; tslat[1][2]++){

		  //copy the superlattice vectors into lattice_point objects
		  //and get their cartesian coordinates

		  vec lat_point[3];

		  for(i=0; i<3; i++){
		    for(j=0; j<3; j++) lat_point[i].fcoord[j]=tslat[i][j];
		    lat_point[i].frac_on=true;
		    conv_AtoB(FtoC, lat_point[i].fcoord, lat_point[i].ccoord);
		    lat_point[i].cart_on=true;
		  }

		  //if no supercells have been added for this volume, add it
		  //else perform all point group operations to the supercell and
		  //see if it is a linear combination of already found supercells with the same volume

		  if(tsupercell.size() == 0){
		    structure tsup_lat;
		    strcpy(tsup_lat.title,"supercell of ");
		    int leng=strlen(tsup_lat.title);
		    for(i=0; title[i]!=0 && i<199-leng; i++)tsup_lat.title[i+leng]=title[i];
		    tsup_lat.scale = scale;
		    for(i=0; i<3; i++){
		      for(j=0; j<3; j++){
			tsup_lat.lat[i][j]= lat_point[i].ccoord[j];
			tsup_lat.slat[i][j]= lat_point[i].fcoord[j];
		      }
		    }

		    tsup_lat.get_trans_mat();
		    tsupercell.push_back(tsup_lat);


		  }
		  else{
		    //apply each point group to this superlattice stored in lat_point

		    int num_miss1=0;
		    for(pg=0; pg<point_group.size(); pg++){
		      vec tlat_point[3];

		      for(i=0; i<3; i++) tlat_point[i]=lat_point[i].apply_sym(point_group[pg]);

		      //see if tlat_point[] can be expressed as a linear combination of any
		      //superlattice with volume vol already found

		      int num_miss2=0;
		      for(int ts=0; ts<tsupercell.size(); ts++){
			double lin_com[3][3];    //contains the coefficients relating the two lattices
			for(i=0; i<3; i++){
			  for(j=0; j<3; j++){
			    lin_com[i][j]=0.0;
			    for(k=0; k<3; k++)
			      lin_com[i][j]=lin_com[i][j]+tsupercell[ts].CtoF[i][k]*tlat_point[j].ccoord[k];
			  }
			}

			//check whether lin_com[][] are strictly integer
			//if so, the transformed superlattice is a linear combination of a previous one

			if(!is_integer(lin_com))num_miss2++;
		      }
		      if(num_miss2 == tsupercell.size())num_miss1++;
		    }
		    if(num_miss1 == point_group.size()){
		      structure tsup_lat;
		      strcpy(tsup_lat.title,"supercell of ");
		      int leng=strlen(tsup_lat.title);
		      for(i=0; title[i]!=0 && i<199-leng; i++)tsup_lat.title[i+leng]=title[i];
		      tsup_lat.scale = scale;
		      for(i=0; i<3; i++){
			for(j=0; j<3; j++){
			  tsup_lat.lat[i][j]= lat_point[i].ccoord[j];
			  tsup_lat.slat[i][j]= lat_point[i].fcoord[j];
			}
		      }
		      tsup_lat.get_trans_mat();
		      tsupercell.push_back(tsup_lat);

		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    for(i=0; i<tsupercell.size(); i++)supercell.push_back(tsupercell[i]);
  }
  return;
}



//************************************************************
void structure::generate_2d_supercells(vector<structure> &supercell, int max_vol, int excluded_axis){
  int vol,pg,i,j,k;
  int tslat[3][3];
  int tslat2d[2][2];


  //algorithm relayed to me by Tim Mueller
  //make upper triangular matrix where the product of the diagonal elements equals the volume
  //then for the elements above the diagonal choose all values less than the diagonal element
  //for each lattice obtained this way, apply point group symmetry operations
  //see if the transformed superlattice can be written as a linear combination of superlattices already found
  //if not add it to the list


  for(vol = 1; vol <= max_vol; vol++){
    vector<structure> tsupercell;


    //generate all tslat[][] matrices that are upper diagonal where the product of the
    //diagonal equals the current volume vol and where the upper diagonal elements take
    //all values less than the diagonal below it

    //initialize the superlattice vectors to zero
    for(i=0; i<3; i++)
      for(j=0; j<3; j++)tslat[i][j]=0;

    for(tslat2d[0][0]=1; tslat2d[0][0]<=vol; tslat2d[0][0]++){
      if(vol%tslat2d[0][0] == 0){
	tslat2d[1][1]=(vol/tslat2d[0][0]);
	tslat2d[1][0]=0;
	for(tslat2d[0][1]=0; tslat2d[0][1]<tslat2d[1][1]; tslat2d[0][1]++){

	  if(excluded_axis=0){
	    tslat[0][0]=1; tslat[0][1]=0; tslat[0][2]=0;
	    tslat[1][0]=0; tslat[1][1]=tslat2d[0][0]; tslat[1][2]=tslat2d[0][1];
	    tslat[2][0]=0; tslat[2][1]=tslat2d[1][0]; tslat[2][2]=tslat2d[1][1];
	  }
	  if(excluded_axis=1){
	    tslat[0][0]=tslat2d[0][0]; tslat[0][1]=0; tslat[0][2]=tslat2d[0][1];
	    tslat[1][0]=0; tslat[1][1]=1; tslat[1][2]=0;
	    tslat[2][0]=tslat2d[1][0]; tslat[2][1]=0; tslat[2][2]=tslat2d[1][1];
	  }
	  if(excluded_axis=2){
	    tslat[0][0]=tslat2d[0][0]; tslat[0][1]=tslat2d[0][1]; tslat[0][2]=0;
	    tslat[1][0]=tslat2d[1][0]; tslat[1][1]=tslat2d[1][1]; tslat[1][2]=0;
	    tslat[2][0]=0; tslat[2][1]=0; tslat[2][2]=1;
	  }

	  //copy the superlattice vectors into lattice_point objects
	  //and get their cartesian coordinates

	  vec lat_point[3];

	  for(i=0; i<3; i++){
	    for(j=0; j<3; j++) lat_point[i].fcoord[j]=tslat[i][j];
	    lat_point[i].frac_on=true;
	    conv_AtoB(FtoC, lat_point[i].fcoord, lat_point[i].ccoord);
	    lat_point[i].cart_on=true;
	  }

	  //if no supercells have been added for this volume, add it
	  //else perform all point group operations to the supercell and
	  //see if it is a linear combination of already found supercells with the same volume


	  if(tsupercell.size() == 0){
	    structure tsup_lat;
	    tsup_lat.scale = scale;
	    for(i=0; i<3; i++){
	      for(j=0; j<3; j++){
		tsup_lat.lat[i][j]= lat_point[i].ccoord[j];
		tsup_lat.slat[i][j]= lat_point[i].fcoord[j];
	      }
	    }
	    tsup_lat.get_trans_mat();
	    tsupercell.push_back(tsup_lat);
	  }
	  else{
	    //apply each point group to this superlattice stored in lat_point

	    int num_miss1=0;
	    for(pg=0; pg<point_group.size(); pg++){
	      vec tlat_point[3];

	      for(i=0; i<3; i++) tlat_point[i]=lat_point[i].apply_sym(point_group[pg]);

	      //see if tlat_point[] can be expressed as a linear combination of any
	      //superlattice with volume vol already found

	      int num_miss2=0;
	      for(int ts=0; ts<tsupercell.size(); ts++){
		double lin_com[3][3];    //contains the coefficients relating the two lattices
		for(i=0; i<3; i++){
		  for(j=0; j<3; j++){
		    lin_com[i][j]=0.0;
		    for(k=0; k<3; k++)
		      lin_com[i][j]=lin_com[i][j]+tsupercell[ts].CtoF[i][k]*tlat_point[j].ccoord[k];
		  }
		}

		//check whether lin_com[][] are strictly integer
		//if so, the transformed superlattice is a linear combination of a previous one

		if(!is_integer(lin_com))num_miss2++;
	      }
	      if(num_miss2 == tsupercell.size())num_miss1++;
	    }
	    if(num_miss1 == point_group.size()){
	      structure tsup_lat;
	      tsup_lat.scale = scale;
	      for(i=0; i<3; i++){
		for(j=0; j<3; j++){
		  tsup_lat.lat[i][j]= lat_point[i].ccoord[j];
		  tsup_lat.slat[i][j]= lat_point[i].fcoord[j];
		}
	      }
	      tsup_lat.get_trans_mat();
	      tsupercell.push_back(tsup_lat);

	    }
	  }
	}
      }
    }
    for(i=0; i<tsupercell.size(); i++)supercell.push_back(tsupercell[i]);
  }
  return;
}
//************************************************************

void structure::generate_lat(structure prim){   /// added by jishnu

  //lat[][]=slat[][]*prim.lat[][]    in matrix form
  //this routine determines lat[][] given slat and prim.lat
  //then it rounds all elements of lat[][] to the nearest integer

  //


  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      lat[i][j]=0.0;
      for(int k=0; k<3; k++){
	lat[i][j]=lat[i][j]+slat[i][k]*prim.lat[k][j]*prim.scale;
      }
    }
  }
  scale=1.0;

}    // end of s/r



//************************************************************

void structure::generate_slat(structure prim){

  //lat[][]=slat[][]*prim.lat[][]    in matrix form
  //this routine determines slat[][]
  //then it rounds all elements of slat[][] to the nearest integer



  double inv_lat[3][3];
  inverse(prim.lat,inv_lat);

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      slat[i][j]=0.0;
      for(int k=0; k<3; k++){
	slat[i][j]=slat[i][j]+lat[i][k]*inv_lat[k][j];
      }
    }
  }

  //round the elements of slat[][] to the closest integer

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      if(slat[i][j]-floor(slat[i][j]) < 0.5) slat[i][j]=floor(slat[i][j]);
      else slat[i][j]=ceil(slat[i][j]);
    }
  }

}


//************************************************************

void structure::generate_slat(structure prim, double rescale){

  //lat[][]=slat[][]*prim.lat[][]    in matrix form
  //this routine determines slat[][]
  //then it rounds all elements of slat[][] to the nearest integer


  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      lat[i][j]=rescale*lat[i][j];
    }

  double inv_lat[3][3];
  inverse(prim.lat,inv_lat);

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      slat[i][j]=0.0;
      for(int k=0; k<3; k++){
	slat[i][j]=slat[i][j]+lat[i][k]*inv_lat[k][j];
      }
    }
  }

  //round the elements of slat[][] to the closest integer

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      if(slat[i][j]-floor(slat[i][j]) < 0.5) slat[i][j]=floor(slat[i][j]);
      else slat[i][j]=ceil(slat[i][j]);
    }
  }


}


//************************************************************

void structure::generate_ideal_slat(structure prim){

  //ilat[][]=slat[][]*prim.lat[][]    in matrix form
  //this routine determines slat[][]
  //then it rounds all elements of slat[][] to the nearest integer

  double inv_lat[3][3];
  inverse(prim.lat,inv_lat);

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      slat[i][j]=0.0;
      for(int k=0; k<3; k++){
	slat[i][j]=slat[i][j]+ilat[i][k]*inv_lat[k][j];
      }
    }
  }

  //round the elements of slat[][] to the closest integer

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      if(slat[i][j]-floor(slat[i][j]) < 0.5) slat[i][j]=floor(slat[i][j]);
      else slat[i][j]=ceil(slat[i][j]);
    }
  }


}


//************************************************************

void structure::generate_ideal_slat(structure prim, double rescale){

  //ilat[][]=slat[][]*prim.lat[][]    in matrix form
  //this routine determines slat[][]
  //then it rounds all elements of slat[][] to the nearest integer

  double tilat[3][3];

  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++){
      tilat[i][j]=rescale*ilat[i][j];
    }

  double inv_lat[3][3];
  inverse(prim.lat,inv_lat);

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      slat[i][j]=0.0;
      for(int k=0; k<3; k++){
	slat[i][j]=slat[i][j]+tilat[i][k]*inv_lat[k][j];
      }
    }
  }

  //round the elements of slat[][] to the closest integer

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      if(slat[i][j]-floor(slat[i][j]) < 0.5) slat[i][j]=floor(slat[i][j]);
      else slat[i][j]=ceil(slat[i][j]);
    }
  }


}

//************************************************************

void structure::calc_strain(){

  //get the matrix the relates lat[][]^transpose=deform[][]*ilat[][]^transpose
  //get the symmetric part of deform as 1/2(deform[][]+deform[][]^transpose)

  double tilat[3][3],tlat[3][3],inv_tilat[3][3],deform[3][3];

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      tlat[i][j]=lat[j][i];
      tilat[i][j]=ilat[j][i];
    }
  }

  inverse(tilat,inv_tilat);

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      deform[i][j]=0.0;
      for(int k=0; k<3; k++)
	deform[i][j]=deform[i][j]+tlat[i][k]*inv_tilat[k][j];
    }
    //    deform[i][i]=deform[i][i]-1.0;
  }

  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++) strain[i][j]=0.5*(deform[i][j]+deform[j][i]);


}

//************************************************************

void structure::generate_prim_grid(){
  int i,j,k;

  prim_grid.clear();

  //create a mesh of primitive lattice points that encompasses the supercell

  //first determine the extrema of all corners of the supercell projected onto the
  //different axes of the primitive cell

  int min[3],max[3];
  int corner[8][3];

  //generate the corners of the supercell

  for(j=0; j<3; j++) corner[0][j]=0;
  for(i=0; i<3; i++)
    for(j=0; j<3; j++){
      if(slat[i][j] <= 0.0) corner[i+1][j]=int(floor(slat[i][j]));
      if(slat[i][j] > 0.0) corner[i+1][j]=int(ceil(slat[i][j]));
    }

  //add up pairs of lattice vectors
  for(i=0; i<3; i++)
    for(j=0; j<3; j++){
      if(slat[(i+1)%3][j]+slat[(i+2)%3][j] <= 0.0) corner[i+4][j]=int(floor(slat[(i+1)%3][j]+slat[(i+2)%3][j]));
      if(slat[(i+1)%3][j]+slat[(i+2)%3][j] > 0.0) corner[i+4][j]=int(ceil(slat[(i+1)%3][j]+slat[(i+2)%3][j]));
    }

  //add up all three lattice vectors
  for(j=0; j<3; j++){
    if(slat[0][j]+slat[1][j]+slat[2][j] <= 0.0) corner[7][j]=int(floor(slat[0][j]+slat[1][j]+slat[2][j]));
    if(slat[0][j]+slat[1][j]+slat[2][j] > 0.0) corner[7][j]=int(ceil(slat[0][j]+slat[1][j]+slat[2][j]));
  }


  //get the extrema of the coordinates projected on the primitive

  for(j=0; j<3; j++){
    min[j]=corner[0][j];
    max[j]=corner[0][j];
  }

  for(i=1; i<8; i++){
    for(j=0; j<3; j++){
      if(min[j] > corner[i][j]) min[j]=corner[i][j];
      if(max[j] < corner[i][j]) max[j]=corner[i][j];
    }
  }


  //generate a grid of primitive lattice sites that encompasses the supercell
  //keep only those primitive lattice sites that reside within the supercell

  for(i=min[0]; i <= max[0]; i++){
    for(j=min[1]; j <= max[1]; j++){
      for(k=min[2]; k <= max[2]; k++){
	double ptemp[3],stemp[3],ctemp[3];

	ptemp[0]=i;
	ptemp[1]=j;
	ptemp[2]=k;

	conv_AtoB(PtoS,ptemp,stemp);

	int ll=0;
	for(int ii=0; ii<3; ii++)
	  if(stemp[ii] >=0.0-tol && stemp[ii] < 1.0-tol) ll++;

	if(ll == 3){
	  vec tlat_point;
	  conv_AtoB(FtoC,stemp,ctemp);
	  for(int jj=0; jj<3; jj++){
	    tlat_point.fcoord[jj]=stemp[jj];
	    tlat_point.ccoord[jj]=ctemp[jj];
	  }
	  tlat_point.frac_on=true;
	  tlat_point.cart_on=true;

	  prim_grid.push_back(tlat_point);
	}
      }

    }
  }

}


//************************************************************
//algorithm taken from B. Z. Yanchitsky, A. N. Timoshevskii,
//Computer Physics Communications vol 139 (2001) 235-242


void structure::generate_3d_reduced_cell(){
  int i,j,k;
  double rlat[3][3];
  double rslat[3][3];
  double leng[3],angle[3];

  for(i=0; i<3; i++)
    for(j=0; j<3; j++){
      rlat[i][j]=lat[i][j];
      rslat[i][j]=slat[i][j];
    }

  double vol=determinant(lat);
  latticeparam(lat,leng,angle);

  //get the largest lattice parameter
  //for all diagonals, find that smallest one that reduces the cell
  //use that diagonal to reduce the cell
  //if among all the diagonals none are smaller than the existing lattice vectors, the cell is reduced
  //then put it into Niggli unique form

  bool small_diag=true;
  while(small_diag){

    int replace;
    double min_leng=leng[0];
    double min_diag[3],min_sdiag[3];

    for(i=0; i<3; i++) if(min_leng < leng[i])min_leng=leng[i];

    small_diag=false;
    for(int i0=-1; i0<=1; i0++){
      for(int i1=-1; i1<=1; i1++){
	for(int i2=-1; i2<=1; i2++){
	  if(!(i0==0 && i1==0) && !(i0==0 && i2==0) && !(i1==0 && i2==0)){

	    double diag[3];
	    double sdiag[3];
	    double tleng=0.0;

	    for(j=0; j<3; j++){
	      diag[j]=i0*rlat[0][j]+i1*rlat[1][j]+i2*rlat[2][j];
	      tleng=tleng+diag[j]*diag[j];
	      sdiag[j]=i0*rslat[0][j]+i1*rslat[1][j]+i2*rslat[2][j];
	    }
	    tleng=sqrt(tleng);
	    if(tleng < min_leng){

	      for(i=0; i<3; i++){

		if(tleng < leng[i]-tol){
		  double tlat[3][3],tslat[3][3];
		  for(j=0; j<3; j++)
		    for(k=0; k<3; k++){
		      tlat[j][k]=rlat[j][k];
		      tslat[j][k]=rslat[j][k];
		    }
		  for(k=0; k<3; k++){
		    tlat[i][k]=diag[k];
		    tslat[i][k]=sdiag[k];
		  }

		  if(abs(determinant(tlat)-vol) < tol){
		    min_leng=tleng;
		    replace=i;
		    for(k=0; k<3; k++){
		      min_diag[k]=diag[k];
		      min_sdiag[k]=sdiag[k];
		    }
		    small_diag=true;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    if(small_diag){
      for(k=0; k<3; k++){
	rlat[replace][k]=min_diag[k];
	rslat[replace][k]=min_sdiag[k];
      }
      latticeparam(rlat,leng,angle);
    }
  }

  //rearrange cell so a < b < c

  int rpermut[3];
  latticeparam(rlat,leng,angle,rpermut);
  for(i=0; i<3; i++)
    for(j=0; j<3; j++){
      lat[i][j]=rlat[rpermut[i]][j];
      slat[i][j]=rslat[rpermut[i]][j];
    }

  //make sure that the angles are either all obtuse or all acute.
  bool found=false;
  for(int i0=1; i0>=-1; i0--){
    for(int i1=1; i1>=-1; i1--){
      for(int i2=1; i2>=-1; i2--){
	if(!found && (i0!=0) && (i1!=0) && (i2!=0)){
	  for(j=0; j<3; j++){
	    rlat[0][j]=i0*lat[0][j];
	    rslat[0][j]=i0*slat[0][j];
	    rlat[1][j]=i1*lat[1][j];
	    rslat[1][j]=i1*slat[1][j];
	    rlat[2][j]=i2*lat[2][j];
	    rslat[2][j]=i2*slat[2][j];
	  }
	  if(determinant(rlat) > 0){
	    latticeparam(rlat,leng,angle);
	    if(abs(angle[0]-90) < tol && abs(angle[1]-90) < tol && abs(angle[2]-90) < tol) found=true;
	    if(angle[0]-90 > -tol && angle[1]-90 > -tol && angle[2]-90 > -tol) found=true;
	    if(angle[0]-90 < tol && angle[1]-90 < tol && angle[2]-90 < tol) found=true;
	  }
	}
      }
    }
  }

  for(i=0; i<3; i++)
    for(j=0; j<3; j++){
      lat[i][j]=rlat[i][j];
      slat[i][j]=rslat[i][j];
    }

  //put the reduced cell into Niggli form

}


//************************************************************
//algorithm taken from B. Z. Yanchitsky, A. N. Timoshevskii,
//Computer Physics Communications vol 139 (2001) 235-242


void structure::generate_2d_reduced_cell(int excluded_axis){
  int i,j,k;
  double rlat[3][3];
  double rslat[3][3];
  double leng[3],angle[3];

  for(i=0; i<3; i++)
    for(j=0; j<3; j++){
      rlat[i][j]=lat[i][j];
      rslat[i][j]=slat[i][j];
    }

  double vol=determinant(lat);
  latticeparam(lat,leng,angle);

  //get the largest lattice parameter
  //for all diagonals, find that smallest one that reduces the cell
  //use that diagonal to reduce the cell
  //if among all the diagonals none are smaller than the existing lattice vectors, the cell is reduced
  //then put it into Niggli unique form

  bool small_diag=true;
  while(small_diag){

    int replace;
    double min_leng=0;
    double min_diag[3],min_sdiag[3];

    for(i=0; i<3; i++)
      if(i != excluded_axis)
	if(min_leng < leng[i])min_leng=leng[i];

    small_diag=false;
    for(int ii0=-1; ii0<=1; ii0++){
      for(int ii1=-1; ii1<=1; ii1++){
	for(int ii2=-1; ii2<=1; ii2++){
	  int i0=ii0;
	  int i1=ii1;
	  int i2=ii2;
	  if(excluded_axis == 0) i0=0;
	  if(excluded_axis == 1) i1=0;
	  if(excluded_axis == 2) i2=0;

	  if(!(i0==0 && i1==0) && !(i0==0 && i2==0) && !(i1==0 && i2==0)){

	    double diag[3];
	    double sdiag[3];
	    double tleng=0.0;

	    for(j=0; j<3; j++){
	      diag[j]=i0*rlat[0][j]+i1*rlat[1][j]+i2*rlat[2][j];
	      tleng=tleng+diag[j]*diag[j];
	      sdiag[j]=i0*rslat[0][j]+i1*rslat[1][j]+i2*rslat[2][j];
	    }
	    tleng=sqrt(tleng);
	    if(tleng < min_leng){

	      for(i=0; i<3; i++){
		if(i != excluded_axis){
		  if(tleng < leng[i]-tol){
		    double tlat[3][3],tslat[3][3];
		    for(j=0; j<3; j++)
		      for(k=0; k<3; k++){
			tlat[j][k]=rlat[j][k];
			tslat[j][k]=rslat[j][k];
		      }
		    for(k=0; k<3; k++){
	              tlat[i][k]=diag[k];
		      tslat[i][k]=sdiag[k];
		    }

		    if(abs(determinant(tlat)-vol) < tol){
		      min_leng=tleng;
		      replace=i;
		      for(k=0; k<3; k++){
			min_diag[k]=diag[k];
			min_sdiag[k]=sdiag[k];
		      }
		      small_diag=true;
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    if(small_diag){
      for(k=0; k<3; k++){
	rlat[replace][k]=min_diag[k];
	rslat[replace][k]=min_sdiag[k];
      }
      latticeparam(rlat,leng,angle);
    }
  }

  for(i=0; i<3; i++)
    for(j=0; j<3; j++){
      lat[i][j]=rlat[i][j];
      slat[i][j]=rslat[i][j];
    }

}



//************************************************************

void specie::print(ostream &stream){
  //for(int i=0; i<2; i++) stream << name[i];   // commented by jishnu
  stream << name;   // jishnu
  return;
}



//************************************************************

atompos::atompos(){

  bit=0;
  specie tcompon;
  compon.push_back(tcompon);

  for(int i=0; i<3; i++){
    fcoord[i]=0.0;
    ccoord[i]=0.0;
    dfcoord[i]=0.0;
    dccoord[i]=0.0;
  }
}


//************************************************************

atompos atompos::apply_sym(sym_op op){
  int i,j,k;
  atompos tatom;

  if(op.frac_on == false && op.cart_on == false){
    cout << "no coordinates available to perform a symmetry operation on\n";
    return tatom;
  }

  if(op.frac_on == false || op.cart_on == false)op.update();

  tatom.bit=bit;
  tatom.occ=occ;

  // added by anton
  for(i=0; i< compon.size(); i++) tatom.compon.push_back(compon[i]);
  for(i=0; i< p_vec.size(); i++) tatom.p_vec.push_back(p_vec[i]);
  for(i=0; i< spin_vec.size(); i++) tatom.spin_vec.push_back(spin_vec[i]);
  for(i=0; i< basis_vec.size(); i++) tatom.basis_vec.push_back(basis_vec[i]);
  tatom.basis_flag=basis_flag;

  tatom.compon.clear();
  for(i=0; i<compon.size(); i++) tatom.compon.push_back(compon[i]);

  for(i=0; i<3; i++){
    tatom.fcoord[i]=op.ftau[i];
    tatom.ccoord[i]=op.ctau[i];
    for(j=0; j<3; j++){
      tatom.fcoord[i]=tatom.fcoord[i]+op.fsym_mat[i][j]*fcoord[j];
      tatom.ccoord[i]=tatom.ccoord[i]+op.csym_mat[i][j]*ccoord[j];
    }
  }
  return tatom;
}


//************************************************************

void atompos::get_cart(double FtoC[3][3]){
  for(int i=0; i<3; i++){
    ccoord[i]=0.0;
    for(int j=0; j<3; j++){
      ccoord[i]=ccoord[i]+FtoC[i][j]*fcoord[j];
    }
  }
}



//************************************************************

void atompos::get_frac(double CtoF[3][3]){
  for(int i=0; i<3; i++){
    fcoord[i]=0.0;
    for(int j=0; j<3; j++){
      fcoord[i]=fcoord[i]+CtoF[i][j]*ccoord[j];
    }
  }
}



//************************************************************

void atompos::readf(istream &stream){

  for(int i=0; i<3; i++) stream >> fcoord[i];

  bit=0;   // is set to zero, unless we read a different number after the specie list

  char buff[200];
  char sp[200];
  stream.getline(buff,199);

  int on=0;
  int off=1;
  int count=0;
  int ii;
  compon.clear();   //clear the blanck specie in the component vector
  for(ii=0; buff[ii]!=0; ii++){
    if(buff[ii] != ' ' && buff[ii] != '\t'){
      if(off == 1){
	on=1;
	off=0;
	count=0;
	for(int jj=0; jj< sizeof(sp); jj++) sp[jj]=' ';
	sp[count]=buff[ii];
      }
      else{
	count++;
	sp[count]=buff[ii];

      }
    }
    else{
      if(on == 1){
	on=0;
	off=1;
	if(isdigit(sp[0])){
	  bit=atoi(sp);
	}
	else{
	  specie tcompon;
	  //for(int jj=0; jj<2; jj++) tcompon.name[jj]=sp[jj];
	  int jj=0;
	  do{
	    tcompon.name.push_back(sp[jj]);
	    jj++;
	  }while(!(sp[jj]==' ') && !(sp[jj]=='\n'));
	  compon.push_back(tcompon);
	}
      }
    }
  }
  if(buff[ii] == 0 && on == 1){
    on=0;
    off=1;
    if(isdigit(sp[0])){
      bit=atoi(sp);
    }
    else{
      specie tcompon;
      //for(int jj=0; jj<2; jj++) tcompon.name[jj]=sp[jj];
      int jj=0;
      do{
	tcompon.name.push_back(sp[jj]);
	jj++;
      }while(!(sp[jj]==' ') && !(sp[jj]=='\n'));
      compon.push_back(tcompon);
    }
  }

}


//************************************************************

void atompos::readc(istream &stream){
  for(int i=0; i<3; i++) stream >> ccoord[i];

  bit=0;   // is set to zero, unless we read a different number after the specie list

  char buff[200];
  char sp[200];
  stream.getline(buff,199);

  int on=0;
  int off=1;
  int count=0;
  int ii;
  compon.clear();   //clear the blanck specie in the component vector
  for(ii=0; buff[ii]!=0; ii++){
    if(buff[ii] != ' ' && buff[ii] != '\t'){
      if(off == 1){
	on=1;
	off=0;
	count=0;
	for(int jj=0; jj< sizeof(sp); jj++) sp[jj]=' ';
	sp[count]=buff[ii];
      }
      else{
	count++;
	sp[count]=buff[ii];

      }
    }
    else{
      if(on == 1){
	on=0;
	off=1;
	if(isdigit(sp[0])){
	  bit=atoi(sp);
	}
	else{
	  specie tcompon;
	  //for(int jj=0; jj<2; jj++) tcompon.name[jj]=sp[jj];
	  int jj=0;
	  do{
	    tcompon.name.push_back(sp[jj]);
	    jj++;
	  }while(!(sp[jj]==' ') && !(sp[jj]=='\n'));
	  compon.push_back(tcompon);
	}
      }
    }
  }
  if(buff[ii] == 0 && on == 1){
    on=0;
    off=1;
    if(isdigit(sp[0])){
      bit=atoi(sp);
    }
    else{
      specie tcompon;
      //for(int jj=0; jj<2; jj++) tcompon.name[jj]=sp[jj];
      int jj=0;
      do{
	tcompon.name.push_back(sp[jj]);
	jj++;
      }while(!(sp[jj]==' ') && !(sp[jj]=='\n'));
      compon.push_back(tcompon);
    }
  }

}



//************************************************************

void atompos::print(ostream &stream){
  for(int i=0; i<3; i++){
    stream.precision(5);stream.width(10);stream.setf(ios::showpoint);
    stream << fcoord[i] << "  ";
  }
  for(int i=0; i<compon.size(); i++){
    //for(int j=0; j<2; j++) stream << compon[i].name[j]; // commented by jishnu
    stream << compon[i].name;   // jishnu
    stream << "  ";
  }
  stream<< bit << " ";
  stream << "\n";
}


//************************************************************

void atompos::assign_spin(){

  int remain=compon.size()%2;
  int bound=(compon.size()-remain)/2;

  int j=bound;
  for(int i=0; i<compon.size(); i++){
    if(j==0 && remain == 0) j--;
    compon[i].spin=j;
    j--;
  }

  return;
}


//************************************************************

void atompos::assemble_flip(){
  for(int i=0; i<compon.size(); i++){
    vector<int> tflip;
    for(int j=1; j<compon.size(); j++){
      tflip.push_back(compon[(i+j)%compon.size()].spin);
    }
    flip.push_back(tflip);
  }
}


//************************************************************

int atompos::iflip(int spin){
  int i;
  for(i=0; i<compon.size(); i++){
    if(compon[i].spin == spin) return i;
  }
  cout << "spin = " << spin << "  is not among those for this sublattice \n";
}

//************************************************************

int atompos::get_spin(string name){
  for(int i=0; i<compon.size(); i++){
    if(compon[i].name.compare(name) == 0) return compon[i].spin;
  }
  //cout << name[0] << name[1] << " is not among the list of components\n";   // commented by jishnu
  cout << name << " is not among the list of components\n";

}




//************************************************************

void structure::bring_in_cell(){
  for(int na=0; na<atom.size(); na++)within(atom[na].fcoord);
  calc_cartesian();
  return;
}



//************************************************************

void structure::calc_factor_group(){
  int pg,i,j,k,n,m,num_suc_maps;
  atompos hatom;
  double shift[3],temp[3];
  sym_op tfactor_group;



  //all symmetry operations are done within the fractional coordinate system
  //since translations back into the unit cell are straightforward

  //for each point group operation, apply it to the crystal and store the transformed
  //coordinates in tatom[]


  for(pg=0; pg<point_group.size(); pg++){
    vector<atompos> tatom;
    for(i=0; i<atom.size(); i++){
      hatom=atom[i].apply_sym(point_group[pg]);
      tatom.push_back(hatom);
    }


    //consider all internal shifts that move an atom of the original structure (e.g. the first
    //atom) to an atom of the transformed structure. Then see if that translation maps the
    //transformed crystal onto the original crystal.


    for(i=0; i<atom.size(); i++){
      if(compare(atom[0].compon, atom[i].compon)){
	for(j=0; j<3; j++) shift[j]=atom[0].fcoord[j]-tatom[i].fcoord[j];

	num_suc_maps=0;
	for(n=0; n<atom.size(); n++){
	  for(m=0; m<atom.size(); m++){
	    if(compare(atom[n].compon, atom[m].compon)){
	      for(j=0; j<3; j++) temp[j]=atom[n].fcoord[j]-tatom[m].fcoord[j]-shift[j];
	      within(temp);

	      k=0;
	      for(j=0; j<3; j++)
		if(abs(temp[j]) < 0.00005 ) k++;
	      if(k==3)num_suc_maps++;
	    }
	  }
	}

	if(num_suc_maps == atom.size()){
	  within(shift);

	  //check whether the symmetry operation already exists in the factorgroup array

	  int ll=0;
	  for(int ii=0; ii<factor_group.size(); ii++)
	    if(compare(point_group[pg].fsym_mat,factor_group[ii].fsym_mat)
	       && compare(shift,factor_group[ii].ftau) )break;
	    else ll++;

	  // if the symmetry operation is new, add it to the factorgroup array
	  // and update all info about the sym_op object

	  if(factor_group.size() == 0 || ll == factor_group.size()){
	    for(int jj=0; jj<3; jj++){
	      tfactor_group.frac_on=false;
	      tfactor_group.cart_on=false;
	      tfactor_group.ftau[jj]=shift[jj];
	      for(int kk=0; kk<3; kk++){
		tfactor_group.fsym_mat[jj][kk]=point_group[pg].fsym_mat[jj][kk];
		tfactor_group.lat[jj][kk]=lat[jj][kk];
	      }
	    }
	    tfactor_group.frac_on=true;
	    tfactor_group.update();
	    tfactor_group.get_sym_type(); // Added by John
	    factor_group.push_back(tfactor_group);
	  }
	}
      }
    }
    tatom.clear();
  }

  return;
}




//************************************************************

void structure::expand_prim_basis(structure prim){
  int i,j,k;


  //add all the prim_grid lattice points to the basis within prim
  //transform the coordinates using PtoS to get the fractional coordinates within
  //current lattice frame

  get_trans_mat();
  generate_prim_grid();

  num_each_specie.clear();
  int ii=0;
  for(i=0; i<prim.num_each_specie.size(); i++){
    int tnum_each_specie=0;


    for(j=0; j<prim.num_each_specie[i]; j++){

      for(k=0; k<prim_grid.size(); k++){
	atompos hatom;
	hatom=prim.atom[ii];
	double temp[3];
	for(int jj=0; jj<3; jj++){
	  temp[jj]=0;
	  for(int kk=0; kk<3; kk++) temp[jj]=temp[jj]+PtoS[jj][kk]*hatom.fcoord[kk];
	}
	for(int kk=0; kk<3; kk++) hatom.fcoord[kk]=temp[kk]+prim_grid[k].fcoord[kk];
	within(hatom);
	hatom.assign_spin();
	atom.push_back(hatom);
	tnum_each_specie++;
      }
      ii++;
    }
    num_each_specie.push_back(tnum_each_specie);
  }
  frac_on=1;
  calc_cartesian();
  //  cout << "calc_cartesian inside expand_prim_cell \n";
  return;
}


//************************************************************

void structure::map_on_expanded_prim_basis(structure prim){

  //takes the current structure and maps the atom positions on the ideal positions

  //prim is expanded according to slat[][]: the expanded structure is ideal_struc
  //this means that the ideal coordinates (within ideal_struc) are those as if simply a volume
  //      relaxation had been done, freezing the internal coordinates

  //the atoms of the current structure are then mapped on to those of ideal_struc
  //the deviation from the ideal positions are stored in dfcoord[] and dccoord[] of atom


  structure ideal_struc;

  //copy title, scale, lat and slat into ideal_struc

  for(int i=0; i<200; i++)ideal_struc.title[i]=title[i];
  ideal_struc.scale=scale;
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      ideal_struc.lat[i][j]=lat[i][j];
      ideal_struc.slat[i][j]=slat[i][j];
    }
  }

  //put ideal lattice parameters in ilat
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      ilat[i][j]=0.0;
      for(int k=0; k<3; k++){
	ilat[i][j]=ilat[i][j]+slat[i][k]*prim.lat[k][j];
      }
    }
  }


  // expand the primitive cell within the slat of ideal_struc
  // expand_prim_basis will convert to cartesian in the lat[][] coordinate system (which is relaxed)

  ideal_struc.expand_prim_basis(prim);

  //if you want distances (cartesian coordinates) in the unrelaxed coordinate system
  //  coord_trans_mat(ilat,ideal_struc.FtoC,ideal_struc.CtoF);
  //  ideal_struc.cart_on=0;
  //  ideal_struc.calc_cartesian();


  // make sure that the current structure has cartesian coordinates (i.e. the one with relaxed coordinates)

  get_trans_mat();
  //  coord_trans_mat(ilat,FtoC,CtoF);      //to have cartesian coordinates and therefore distances in the unrelaxed coordinate system
  calc_cartesian();

  //find the positions in ideal_struc that are closest to those in the current structure
  //   if the position is found, increase atom[].bit from 0 to 1
  //make sure that the first component of structure coincides with one of the components in the component list of ideal_struc
  //if so update occ of ideal_struc and put delta in atom of ideal_struc
  //if some ideal_struc positions are not matched - check if there are vacancies in the component list if so, make the occ = Va

  //transcribe the atom objects of ideal_struc into those of the current structure

  for(int na=0; na<atom.size(); na++){
    if(atom[na].compon.size() == 0){
      cout << "the structure file has no atomic labels next to the coordinates\n";
      cout << "exiting \n";
      exit(1);
    }
    int min_ind=-1;
    int comp_ind=-1;
    int min_i,min_j,min_k;
    double min_dist=1.0e10;

    //look for the closest ideal position
    for(int ina=0; ina<ideal_struc.atom.size(); ina++){

      //first make sure that atom[na].compon[0] belongs to one of ideal_struc.atom[ina].compon
      for(int c=0; c<ideal_struc.atom[ina].compon.size(); c++){

	//if(compare(atom[na].compon[0],ideal_struc.atom[ina].compon[c])){

	if(compare(atom[na].occ,ideal_struc.atom[ina].compon[c])){

	  //need to translate atom to all edges of the unit cell and get the minimal distance
	  //to ideal_struc.atom[ina]
	  double tmin_dist=1.0e10;
	  int tmin_i,tmin_j,tmin_k;
	  for(int i=-1; i<=1; i++){
	    for(int j=-1; j<=1; j++){
	      for(int k=-1; k<=1; k++){
		atompos hatom=atom[na];
		hatom.fcoord[0]=atom[na].fcoord[0]+i;
		hatom.fcoord[1]=atom[na].fcoord[1]+j;
		hatom.fcoord[2]=atom[na].fcoord[2]+k;
		conv_AtoB(FtoC,hatom.fcoord,hatom.ccoord);
		double dist=0.0;
		for(int ii=0; ii<3; ii++){
		  dist=dist+(hatom.ccoord[ii]-ideal_struc.atom[ina].ccoord[ii])*(hatom.ccoord[ii]-ideal_struc.atom[ina].ccoord[ii]);
		}
		dist=sqrt(dist);

		if(dist<tmin_dist){
		  tmin_dist=dist;
		  tmin_i=i;
		  tmin_j=j;
		  tmin_k=k;
		}
	      }
	    }
	  }
	  if(tmin_dist<min_dist){
	    min_dist=tmin_dist;
	    min_ind=ina;
	    comp_ind=c;
	    min_i=tmin_i;
	    min_j=tmin_j;
	    min_k=tmin_k;
	  }
	  break;
	}
      }

    }
    if(min_ind >= 0 && ideal_struc.atom[min_ind].bit==0){
      ideal_struc.atom[min_ind].assign_spin();
      ideal_struc.atom[min_ind].occ=ideal_struc.atom[min_ind].compon[comp_ind];
      atom[na].fcoord[0]=atom[na].fcoord[0]+min_i;
      atom[na].fcoord[1]=atom[na].fcoord[1]+min_j;
      atom[na].fcoord[2]=atom[na].fcoord[2]+min_k;
      conv_AtoB(FtoC,atom[na].fcoord,atom[na].ccoord);

      for(int i=0; i<3; i++){
	ideal_struc.atom[min_ind].dfcoord[i]=atom[na].fcoord[i]-ideal_struc.atom[min_ind].fcoord[i];
	ideal_struc.atom[min_ind].dccoord[i]=atom[na].ccoord[i]-ideal_struc.atom[min_ind].ccoord[i];
      }
      ideal_struc.atom[min_ind].delta=min_dist;
      ideal_struc.atom[min_ind].bit=1;    // this means this sites has already been taken
    }
    else {
      if(min_ind == -1){
	cout << "it was not possible to map atom\n";
	atom[na].print(cout);
	cout << "onto the ideal structure\n";
      }
      if(ideal_struc.atom[min_ind].bit == 1){
	cout << "it was not possible to map atom \n";
	atom[na].print(cout);
	cout << "onto the closest ideal position\n";
	ideal_struc.atom[min_ind].print(cout);
	cout << "since this ideal atom position has already been claimed\n";
      }
    }

  }

  // run through ideal_struc to find sites that have not been claimed yet
  // check if these sites can hold vacancies, if so, occ becomes the vacancy
  // if not, there is a problem

  for(int ina=0; ina<ideal_struc.atom.size(); ina++){
    if(ideal_struc.atom[ina].bit == 0){
      for(int c=0; c< ideal_struc.atom[ina].compon.size(); c++){
	if(ideal_struc.atom[ina].compon[c].name[0] == 'V' && ideal_struc.atom[ina].compon[c].name[1] == 'a'){
	  ideal_struc.atom[ina].occ=ideal_struc.atom[ina].compon[c];
	  ideal_struc.atom[ina].delta=0.0;
	  ideal_struc.atom[ina].bit=1;
	  break;
	}
      }
      if(ideal_struc.atom[ina].bit == 0){
	cout << "was not able to map any atom onto this position\n";
	ideal_struc.atom[ina].print(cout);
      }
    }
  }

  //copy the ideal_struc atom positions into the current structure

  num_each_specie.clear();
  atom.clear();


  //for now, we just list the total number of atoms in num_each_specie (later we may
  //modify this part to do the same as is done when structures are generated from scratch

  num_each_specie.push_back(ideal_struc.atom.size());
  for(int ina=0; ina<ideal_struc.atom.size(); ina++){
    atom.push_back(ideal_struc.atom[ina]);
    //assign to atom.bit the index of the component at that site and assign spins to each component

  }

  //modified by anton
  for(int i=0; i<atom.size(); i++) get_basis_vectors(atom[i]);

  return;

}


//************************************************************

void structure::map_on_expanded_prim_basis(structure prim, arrangement &conf){

  //takes the current structure and maps the atom positions on the ideal positions

  //prim is expanded according to slat[][]: the expanded structure is ideal_struc
  //this means that the ideal coordinates (within ideal_struc) are those as if simply a volume
  //      relaxation had been done, freezing the internal coordinates

  //the atoms of the current structure are then mapped on to those of ideal_struc
  //the deviation from the ideal positions are stored in dfcoord[] and dccoord[] of atom


  structure ideal_struc;

  //copy title, scale, lat and slat into ideal_struc

  for(int i=0; i<200; i++)ideal_struc.title[i]=title[i];
  ideal_struc.scale=scale;
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      ideal_struc.lat[i][j]=lat[i][j];
      ideal_struc.slat[i][j]=slat[i][j];
    }
  }

  // expand the primitive cell within the slat of ideal_struc
  // expand_prim_basis will convert to cartesian in the lat[][] coordinate system (which is relaxed)

  ideal_struc.expand_prim_basis(prim);

  //initialize the bit of each atom to be -1 (means has not been mapped onto yet)

  for(int na=0; na<ideal_struc.atom.size(); na++) ideal_struc.atom[na].bit=-1;

  //if you want distances (cartesian coordinates) in the unrelaxed coordinate system
  //  coord_trans_mat(ilat,ideal_struc.FtoC,ideal_struc.CtoF);
  //  ideal_struc.cart_on=0;
  //  ideal_struc.calc_cartesian();


  // make sure that the current structure has cartesian coordinates (i.e. the one with relaxed coordinates)

  get_trans_mat();
  //  coord_trans_mat(ilat,FtoC,CtoF);      //to have cartesian coordinates and therefore distances in the unrelaxed coordinate system
  calc_cartesian();

  //find the positions in ideal_struc that are closest to those in the current structure
  //   if the position is found, change atom[].bit from -1 to the index of the component at that site
  //make sure that the first component of structure coincides with one of the components in the component list of ideal_struc
  //if so update occ of ideal_struc and put delta in atom of ideal_struc
  //if some ideal_struc positions are not matched - check if there are vacancies in the component list if so, make the occ = Va

  //transcribe the atom objects of ideal_struc into those of the current structure

  for(int na=0; na<atom.size(); na++){
    if(atom[na].compon.size() == 0){
      cout << "the structure file has no atomic labels next to the coordinates\n";
      cout << "exiting \n";
      exit(1);
    }
    int min_ind=-1;
    int comp_ind=-1;
    int min_i,min_j,min_k;
    double min_dist=1.0e10;

    //look for the closest ideal position
    for(int ina=0; ina<ideal_struc.atom.size(); ina++){

      //first make sure that atom[na].compon[0] belongs to one of ideal_struc.atom[ina].compon
      for(int c=0; c<ideal_struc.atom[ina].compon.size(); c++){


	if(compare(atom[na].compon[0],ideal_struc.atom[ina].compon[c])){

	  //need to translate atom to all edges of the unit cell and get the minimal distance
	  //to ideal_struc.atom[ina]
	  double tmin_dist=1.0e10;
	  int tmin_i,tmin_j,tmin_k;
	  for(int i=-1; i<=1; i++){
	    for(int j=-1; j<=1; j++){
	      for(int k=-1; k<=1; k++){
		atompos hatom=atom[na];
		hatom.fcoord[0]=atom[na].fcoord[0]+i;
		hatom.fcoord[1]=atom[na].fcoord[1]+j;
		hatom.fcoord[2]=atom[na].fcoord[2]+k;
		conv_AtoB(FtoC,hatom.fcoord,hatom.ccoord);
		double dist=0.0;
		for(int ii=0; ii<3; ii++){
		  dist=dist+(hatom.ccoord[ii]-ideal_struc.atom[ina].ccoord[ii])*(hatom.ccoord[ii]-ideal_struc.atom[ina].ccoord[ii]);
		}
		dist=sqrt(dist);

		if(dist<tmin_dist){
		  tmin_dist=dist;
		  tmin_i=i;
		  tmin_j=j;
		  tmin_k=k;
		}
	      }
	    }
	  }
	  if(tmin_dist<min_dist){
	    min_dist=tmin_dist;
	    min_ind=ina;
	    comp_ind=c;
	    min_i=tmin_i;
	    min_j=tmin_j;
	    min_k=tmin_k;
	  }
	  break;
	}
      }

    }
    if(min_ind >= 0 && ideal_struc.atom[min_ind].bit==-1){
      ideal_struc.atom[min_ind].assign_spin();
      ideal_struc.atom[min_ind].occ=ideal_struc.atom[min_ind].compon[comp_ind];
      ideal_struc.atom[min_ind].bit=comp_ind;   // also indicates that this sites has already been taken
      atom[na].fcoord[0]=atom[na].fcoord[0]+min_i;
      atom[na].fcoord[1]=atom[na].fcoord[1]+min_j;
      atom[na].fcoord[2]=atom[na].fcoord[2]+min_k;
      conv_AtoB(FtoC,atom[na].fcoord,atom[na].ccoord);

      for(int i=0; i<3; i++){
	ideal_struc.atom[min_ind].dfcoord[i]=atom[na].fcoord[i]-ideal_struc.atom[min_ind].fcoord[i];
	ideal_struc.atom[min_ind].dccoord[i]=atom[na].ccoord[i]-ideal_struc.atom[min_ind].ccoord[i];
      }
      ideal_struc.atom[min_ind].delta=min_dist;
    }
    else {
      if(min_ind == -1){
	cout << "it was not possible to map atom\n";
	atom[na].print(cout);
	cout << "onto the ideal structure\n";
      }
      if(ideal_struc.atom[min_ind].bit != -1){
	cout << "it was not possible to map atom \n";
	atom[na].print(cout);
	cout << "onto the closest ideal position\n";
	ideal_struc.atom[min_ind].print(cout);
	cout << "since this ideal atom position has already been claimed\n";
      }
    }

  }

  // run through ideal_struc to find sites that have not been claimed yet
  // check if these sites can hold vacancies, if so, occ becomes the vacancy
  // if not, there is a problem

  for(int ina=0; ina<ideal_struc.atom.size(); ina++){
    if(ideal_struc.atom[ina].bit == -1){
      for(int c=0; c< ideal_struc.atom[ina].compon.size(); c++){
	if(ideal_struc.atom[ina].compon[c].name[0] == 'V' && ideal_struc.atom[ina].compon[c].name[1] == 'a'){
	  ideal_struc.atom[ina].occ=ideal_struc.atom[ina].compon[c];
	  ideal_struc.atom[ina].delta=0.0;
	  ideal_struc.atom[ina].bit=c;
	  break;
	}
      }
      if(ideal_struc.atom[ina].bit == -1){
	cout << "was not able to map any atom onto this position\n";
	ideal_struc.atom[ina].print(cout);
      }
    }
  }

  //copy the ideal_struc atom positions into the current structure

  num_each_specie.clear();
  atom.clear();
  conf.bit.clear();

  //for now, we just list the total number of atoms in num_each_specie (later we may
  //modify this part to do the same as is done when structures are generated from scratch

  num_each_specie.push_back(ideal_struc.atom.size());
  for(int ina=0; ina<ideal_struc.atom.size(); ina++){
    atom.push_back(ideal_struc.atom[ina]);
    conf.bit.push_back(ideal_struc.atom[ina].bit);
  }

  //modified by anton
  for(int i=0; i<atom.size(); i++) get_basis_vectors(atom[i]);

  return;

}



//************************************************************

void structure::idealize(){
  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++) ilat[i][j]=lat[i][j];

  get_latparam();
  get_trans_mat();
  calc_fractional();
  calc_cartesian();
}


//************************************************************

void structure::expand_prim_clust(multiplet basiplet, multiplet &super_basiplet){
  int nm,no,nc,np,ng;
  int i,j,k;

  get_trans_mat();

  generate_prim_grid();

  //transform the cluster coordinates (in prim system) to the current lattice coordinate system
  //add prim_grid points to each cluster and store the new clusters in super_basis

  super_basiplet.orb.push_back(basiplet.orb[0]);


  for(nm=1; nm<basiplet.orb.size(); nm++){
    vector<orbit> torbvec;
    for(no=0; no<basiplet.orb[nm].size(); no++){
      orbit torb1,torb2;
      for(nc=0; nc<basiplet.orb[nm][no].equiv.size(); nc++){
	cluster tclust;
	for(np=0; np<basiplet.orb[nm][no].equiv[nc].point.size(); np++){
	  atompos tpoint=basiplet.orb[nm][no].equiv[nc].point[np];
	  for(i=0; i<3; i++){
	    tpoint.fcoord[i]=0.0;
	    for(j=0; j<3; j++)
	      tpoint.fcoord[i]=tpoint.fcoord[i]+PtoS[i][j]*basiplet.orb[nm][no].equiv[nc].point[np].fcoord[j];
	  }
	  tclust.point.push_back(tpoint);
	}
	torb2.equiv.push_back(tclust);
      }
      for(ng=0; ng<prim_grid.size(); ng++){
	for(nc=0; nc<torb2.equiv.size(); nc++){
	  cluster tclust;
	  for(np=0; np<torb2.equiv[nc].point.size(); np++){
	    atompos tpoint=torb2.equiv[nc].point[np];
	    for(i=0; i<3; i++){
	      tpoint.fcoord[i]=tpoint.fcoord[i]+prim_grid[ng].fcoord[i];
	    }
	    within(tpoint);
	    tclust.point.push_back(tpoint);
	  }
	  torb1.equiv.push_back(tclust);
	}
      }
      torbvec.push_back(torb1);
    }
    super_basiplet.orb.push_back(torbvec);
  }
  return;
}



//************************************************************

void structure::write_factor_group(){
  int fg;

  ofstream out("factor_group");
  if(!out){
    cout << "Cannot open file.\n";
    return;
  }

  out << " number of factor group ops " << factor_group.size() << "\n";
  out << " *** Please Note:  Depending on translation vectors, rotations and mirrors may correspond to screw axes and glide planes.\n";
  for(fg=0; fg<factor_group.size(); fg++){
    out << "factor group operation " << fg << " \n";
    factor_group[fg].print_fsym_mat(out);
    out << "\n";
  }

  out.close();
}


//************************************************************
//read a POSCAR like file and collect all the structure variables
//modified to read PRIM file and determine which basis to use
//modified by Ben Swoboda

void structure::read_struc_prim(istream &stream){
  int i,j;
  char ch;
  char buff[200];
  char sp[200];
  atompos hatom;

  stream.getline(title,199);
  stream >> scale;
  for(i=0; i<3; i++){
    for(j=0; j<3; j++){
      stream >> lat[i][j];
    }
  }

  //normalize the scale to 1.0 and adjust lattice vectors accordingly

  for(i=0; i<3; i++)
    for(j=0; j<3; j++)lat[i][j]=scale*lat[i][j];
  scale=1.0;


  stream.getline(buff,199);
  stream.getline(buff,199);

  //Figure out how many species

  int on=0;
  int off=1;
  int count=0;

  for(i=0; buff[i]!=0; i++){
    if(buff[i] != ' ' && buff[i] != '\t'){
      if(off == 1){
	on=1;
	off=0;
	count=0;
	for(int ii=0; ii< sizeof(sp); ii++) sp[ii]=' ';
	sp[count]=buff[i];
      }
      else{
	count++;
	sp[count]=buff[i];
      }
    }
    else{
      if(on == 1){
	on=0;
	off=1;

	int ll=atoi(sp);
	num_each_specie.push_back(ll);
      }
    }
  }
  if(buff[i] == 0 && on == 1){
    on=0;
    off=1;
    int ll=atoi(sp);
    num_each_specie.push_back(ll);
  }



  int num_atoms=0;
  for(i=0; i<num_each_specie.size(); i++)num_atoms +=num_each_specie[i];

  // fractional coordinates or cartesian

  stream.getline(buff,199);
  i=0;
  while(buff[i] == ' ')i++;

  //first check for selective dynamics

  if(buff[i] == 'S' || buff[i] == 's') {
    stream.getline(buff,199);
    i=0;
    while(buff[i] == ' ')i++;
  }

  frac_on=false; cart_on=false;

  if(buff[i] == 'D' || buff[i] == 'd'){
    frac_on=true;
    cart_on=false;
  }
  else
    if(buff[i] == 'C' || buff[i] == 'c'){
      frac_on=false;
      cart_on=true;
    }
    else{
      cout << "ERROR in input\n";
      cout << "If not using selective dynamics the 7th line should be Direct or Cartesian.\n";
      cout << "Otherwise the 8th line should be Direct or Cartesian \n";
      exit(1);
    }


  //read the coordinates

  if(atom.size() != 0 ){
    cout << "the structure is going to be overwritten";
    atom.clear();
  }



  // The following part written by jishnu to take care of the spaces in prim file (spaces do not matter any more)
  for(i=0; i<num_atoms; i++){
    for(j=0; j<3; j++){
      if(frac_on == true) stream >> hatom.fcoord[j];
      else stream >> hatom.ccoord[j];
    }
    // cout << "atom #" << i << "coordinates are : " <<  hatom.fcoord[0] << "   " <<  hatom.fcoord[1] << "   "<<  hatom.fcoord[2] << "\n";
    // determine the number and name of species that can occupy this atomposition
    hatom.compon.clear();   //clear the blank specie in the component vector
    do{
      if(!(((stream.peek()>='A')&&(stream.peek()<='Z'))||((stream.peek()>='a')&&(stream.peek()<='z')))) stream.get(ch);
      if(((stream.peek()>='A')&&(stream.peek()<='Z'))||((stream.peek()>='a')&&(stream.peek()<='z'))){
	specie tcompon;
	while(!(stream.peek()==' ')) {
	  stream.get(ch);
	  tcompon.name.push_back(ch);
	  if(stream.peek()=='\n') break;
	}
	hatom.compon.push_back(tcompon);
      }
      if((stream.peek()>='0')&&(stream.peek()<='9')){
	if((stream.peek()>='0')&&(stream.peek()<='1')) stream >> hatom.basis_flag;
	else cout << "Check the prim file; there is an invalid basis_flag \n";
      }
    }while(!(stream.peek()=='\n'));
    hatom.occ=hatom.compon[0];
    atom.push_back(hatom);
  }

}


//************************************************************
//read a POSCAR like file and collect all the structure variables

void structure::read_struc_poscar(istream &stream){
  int i,j;
  char ch;
  char buff[200];
  char sp[200];
  atompos hatom;

  stream.getline(title,199);
  stream >> scale;
  for(i=0; i<3; i++){
    for(j=0; j<3; j++){
      stream >> lat[i][j];
    }
  }

  //normalize the scale to 1.0 and adjust lattice vectors accordingly

  for(i=0; i<3; i++)
    for(j=0; j<3; j++)lat[i][j]=scale*lat[i][j];
  scale=1.0;


  stream.getline(buff,199);
  stream.getline(buff,199);

  //Figure out how many species

  int on=0;
  int off=1;
  int count=0;

  for(i=0; buff[i]!=0; i++){
    if(buff[i] != ' ' && buff[i] != '\t'){
      if(off == 1){
	on=1;
	off=0;
	count=0;
	for(int ii=0; ii< sizeof(sp); ii++) sp[ii]=' ';
	sp[count]=buff[i];
      }
      else{
	count++;
	sp[count]=buff[i];
      }
    }
    else{
      if(on == 1){
	on=0;
	off=1;

	int ll=atoi(sp);
	num_each_specie.push_back(ll);
      }
    }
  }
  if(buff[i] == 0 && on == 1){
    on=0;
    off=1;
    int ll=atoi(sp);
    num_each_specie.push_back(ll);
  }



  int num_atoms=0;
  for(i=0; i<num_each_specie.size(); i++)num_atoms +=num_each_specie[i];

  // fractional coordinates or cartesian

  stream.getline(buff,199);
  i=0;
  while(buff[i] == ' ')i++;

  //first check for selective dynamics

  if(buff[i] == 'S' || buff[i] == 's') {
    stream.getline(buff,199);
    i=0;
    while(buff[i] == ' ')i++;
  }

  frac_on=false; cart_on=false;

  if(buff[i] == 'D' || buff[i] == 'd'){
    frac_on=true;
    cart_on=false;
  }
  else
    if(buff[i] == 'C' || buff[i] == 'c'){
      frac_on=false;
      cart_on=true;
    }
    else{
      cout << "ERROR in input\n";
      cout << "If not using selective dynamics the 7th line should be Direct or Cartesian.\n";
      cout << "Otherwise the 8th line should be Direct or Cartesian \n";
      exit(1);
    }


  //read the coordinates

  if(atom.size() != 0 ){
    cout << "the structure is going to be overwritten";
    atom.clear();
  }

  for(i=0; i<num_atoms; i++){
    for(j=0; j<3; j++)
      if(frac_on == true)stream >> hatom.fcoord[j];
      else stream >> hatom.ccoord[j];

    // determine the number and name of species that can occupy this atomposition

    stream.getline(buff,199);

    int on=0;
    int off=1;
    int count=0;
    int ii;
    hatom.compon.clear();   //clear the blanck specie in the component vector
    for(ii=0; buff[ii]!=0; ii++){
      if(buff[ii] != ' ' && buff[ii] != '\t'){
	if(off == 1){
	  on=1;
	  off=0;
	  count=0;
	  for(int jj=0; jj< sizeof(sp); jj++) sp[jj]=' ';
	  sp[count]=buff[ii];
	}
	else{
	  count++;
	  sp[count]=buff[ii];
	}
      }
      else{
	if(on == 1){
	  on=0;
	  off=1;
	  specie tcompon;
	  //for(int jj=0; jj<2; jj++) tcompon.name[jj]=sp[jj];
	  int jj=0;
	  do{
	    tcompon.name.push_back(sp[jj]);
	    jj++;
	  }while(!(sp[jj]==' ') && !(sp[jj]=='\n'));
	  hatom.compon.push_back(tcompon);
	}
      }
    }
    if(buff[ii] == 0 && on == 1){
      on=0;
      off=1;
      specie tcompon;
      //for(int jj=0; jj<2; jj++) tcompon.name[jj]=sp[jj];
      int jj=0;
      do{
	tcompon.name.push_back(sp[jj]);
	jj++;
      }while(!(sp[jj]==' ') && !(sp[jj]=='\n'));
      hatom.compon.push_back(tcompon);
    }
    //assign the first component to the occupation slot
    hatom.occ=hatom.compon[0];
    atom.push_back(hatom);
  }

}


//************************************************************
//************************************************************
//write the structure to a file in POSCAR like format

void structure::write_struc_poscar(ostream &stream) {
  int i,j;


  stream << title <<"\n";

  stream.precision(7);stream.width(12);stream.setf(ios::showpoint);
  stream << scale <<"\n";

  for(int i=0; i<3; i++){
    stream << "  ";
    for(int j=0; j<3; j++){

      stream.precision(9);stream.width(15);stream.setf(ios::showpoint);
      stream << lat[i][j] << "  ";

    }
    stream << "\n";
  }
  for(i=0; i < num_each_specie.size(); i++) stream << " " << num_each_specie[i] ;
  stream << "\n";

  stream << "Direct\n";

  for(i=0; i<atom.size(); i++){
    for(j=0; j<3; j++){
      stream.precision(9);stream.width(15);stream.setf(ios::showpoint);
      stream << atom[i].fcoord[j] << "  ";
    }
    //    for(int ii=0; ii<atom[i].compon.size(); ii++){
    stream << atom[i].occ.name;   // jishnu
    //      stream << "  ";
    //    }
    stream << "\n";
  }
}


//************************************************************
void structure::write_struc_xyz(ostream &stream){
  calc_cartesian();
  int tot_num_atoms=0;
  for(int i=0; i<atom.size(); i++){
    if(!(atom[i].occ.name[0] == 'V' && atom[i].occ.name[1] == 'a'))
      tot_num_atoms++;
  }

  stream << tot_num_atoms << "\n";
  stream << title << "\n";
  for(int i=0; i<atom.size(); i++){
    if(!(atom[i].occ.name[0] == 'V' && atom[i].occ.name[1] == 'a')){
      atom[i].occ.print(stream);
      stream << "  ";
      for(int j=0; j< 3; j++) stream << atom[i].ccoord[j] << "  ";
      stream << "\n";
    }
  }
  return;

}

//************************************************************
void structure::write_struc_xyz(ostream &stream, concentration out_conc){
  calc_cartesian();
  int tot_num_atoms=0;
  for(int i=0; i<atom.size(); i++){
    if(!(atom[i].occ.name[0] == 'V' && atom[i].occ.name[1] == 'a'))
      tot_num_atoms++;
  }

  stream << tot_num_atoms << "\n";
  stream << "Configuration with concentrations: ";
  out_conc.print_concentration(stream);
  stream << "\n";
  for(int i=0; i<atom.size(); i++){
    if(!(atom[i].occ.name[0] == 'V' && atom[i].occ.name[1] == 'a')){
      atom[i].occ.print(stream);
      stream << "  ";
      for(int j=0; j< 3; j++) stream << atom[i].ccoord[j] << "  ";
      stream << "\n";
    }
  }
  return;

}

//************************************************************

void structure::calc_cartesian(){
  int i,j,k;

  if(cart_on == true) return;
  if(cart_on == false) {
    if(frac_on == false){
      cout << "No structure initialized - cannot calc_cartesian\n";
      return;
    }
    for(i=0; i<atom.size(); i++) conv_AtoB(FtoC, atom[i].fcoord, atom[i].ccoord);

    cart_on=true;
  }
}



//************************************************************

void structure::calc_fractional(){
  int i,j,k;

  if(frac_on == true) return;
  if(frac_on == false) {
    if(cart_on == false){
      cout << "No structure initialized - cannot calc_fractional\n";
      return;
    }
    for(i=0; i<atom.size(); i++) conv_AtoB(CtoF, atom[i].ccoord, atom[i].fcoord);

    frac_on=true;
  }
}


//************************************************************

void structure::collect_components(){

  if(atom.size() == 0){
    cout << "cannot collect_components since no atoms in structure \n";
    return;
  }

  //find the first atom that has at least one component
  int i=0;
  while(atom[i].compon.size() < 1 && i<atom.size()){
    i++;
  }
  if(i==atom.size()){
    cout << "no atoms with at least one component\n";
    cout << "not collecting components\n";
    return;
  }


  compon.clear();
  compon.push_back(atom[i].compon[0]);
  for(int i=0; i<atom.size(); i++){
    if(atom[i].compon.size() >= 1){
      for(int j=0; j<atom[i].compon.size(); j++){
        int l=0;
        for(int k=0; k<compon.size(); k++)
          if(!compare(atom[i].compon[j],compon[k]))l++;
        if(l==compon.size())compon.push_back(atom[i].compon[j]);
      }
    }
  }

  return;
}


//************************************************************

void structure::collect_relax(string dir_name){

  //reads a POS and a CONTCAR, collects the info from both and places it
  //all in the current structure object
  //then prints some relevant info in a RELAX file

  //-puts the relaxed cell vectors in rlat (from CONTCAR), puts the original cell vectors in lat (from POS)
  //-puts the unrelaxed coordinates from POS in atompos together with the atom labels
  //-get the difference between the unrelaxed coordinates and the relaxed coordinates (from CONTCAR) and
  //place in dfcoord, dccoord and get the distance delta
  //-after printing out the info about the relaxations in RELAX, replace the atomic coordinates with
  //relaxed coordinates from CONTCAR, set dfcoord, dccoord and delta all to zero.


  //create a string for the POS filename and the CONTCAR filename
  //define structure objects for POS and for CONTCAR

  structure pos;

  string pos_filename=dir_name;
  pos_filename.append("/POS");
  ifstream in_pos;
  in_pos.open(pos_filename.c_str());
  if(!in_pos){
    cout << "cannot open file " << pos_filename << "\n";
    return;
  }

  pos.read_struc_poscar(in_pos);

  in_pos.close();

  structure contcar;

  string contcar_filename=dir_name;
  contcar_filename.append("/CONTCAR");
  ifstream in_contcar;
  in_contcar.open(contcar_filename.c_str());
  if(!in_contcar){
    cout << "cannot open file " << contcar_filename << "\n";
    return;
  }

  contcar.read_struc_poscar(in_contcar);

  in_contcar.close();

  if(pos.atom.size() != contcar.atom.size()){
    cout << "POS and CONTCAR in " << dir_name << "\n";
    cout << "are incompatible \n";
    cout << "quitting collect_relax() \n";
    return;
  }


  //collect the data from the two structures and place it in the current structure

  for(int i=0; i<200; i++) title[i]=pos.title[i];
  scale=pos.scale;

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      ilat[i][j]=pos.lat[i][j];
      lat[i][j]=contcar.lat[i][j];
    }
  }

  atom.clear();

  for(int na=0; na<pos.atom.size(); na++) atom.push_back(pos.atom[na]);
  frac_on=1;

  get_trans_mat();
  calc_cartesian();


  //go through contcar and translate the atoms so they are closest to the pos atom positions
  //we get cartesian coordinates of contcar by transforming the fractional into the
  //relaxed coordinate system

  for(int na=0; na<atom.size(); na++){
    double min_dist=1.0e10;
    int min_i,min_j,min_k;
    for(int i=-1; i<=1; i++){
      for(int j=-1; j<=1; j++){
	for(int k=-1; k<=1; k++){
	  atompos hatom=contcar.atom[na];
	  hatom.fcoord[0]=contcar.atom[na].fcoord[0]+i;
	  hatom.fcoord[1]=contcar.atom[na].fcoord[1]+j;
	  hatom.fcoord[2]=contcar.atom[na].fcoord[2]+k;
	  conv_AtoB(FtoC,hatom.fcoord,hatom.ccoord);
	  double dist=0.0;
	  for(int ii=0; ii<3; ii++)
	    dist=dist+(hatom.ccoord[ii]-atom[na].ccoord[ii])*(hatom.ccoord[ii]-atom[na].ccoord[ii]);
	  dist=sqrt(dist);

	  if(dist<min_dist){
	    min_dist=dist;
	    min_i=i;
	    min_j=j;
	    min_k=k;
	  }
	}
      }
    }
    atompos hatom=contcar.atom[na];
    hatom.fcoord[0]=contcar.atom[na].fcoord[0]+min_i;
    hatom.fcoord[1]=contcar.atom[na].fcoord[1]+min_j;
    hatom.fcoord[2]=contcar.atom[na].fcoord[2]+min_k;
    conv_AtoB(FtoC,hatom.fcoord,hatom.ccoord);

    for(int i=0; i<3; i++){
      atom[na].dfcoord[i]=hatom.fcoord[i]-atom[na].fcoord[i];
      atom[na].dccoord[i]=hatom.ccoord[i]-atom[na].ccoord[i];
    }
    atom[na].delta=min_dist;
  }


  //calculate the strain matrix and the lattice parameters

  calc_strain();
  get_latparam();
  get_ideal_latparam();


  //write some of this info into a RELAX file in the directory containing POS and CONTCAR
  string relax_file=dir_name;
  relax_file.append("/RELAX");
  ofstream out_relax;
  out_relax.open(relax_file.c_str());
  if(!out_relax){
    cout << "cannot open file " << relax_file << "\n";
    return;
  }

  out_relax << "CONFIGURATION = " << dir_name << "\n";
  out_relax << "CHANGE IN VOLUME = " << determinant(strain) << "\n";
  out_relax << "\n";
  out_relax << "Original lattice parameters \n";
  for(int ii=0; ii<3; ii++) out_relax << ilatparam[ii] << "  ";
  for(int ii=0; ii<3; ii++) out_relax << ilatangle[ii] << "  ";
  out_relax << "\n";
  out_relax << "\n";

  out_relax << "Lattice parameters after relaxation \n";
  for(int ii=0; ii<3; ii++) out_relax << latparam[ii] << "  ";
  for(int ii=0; ii<3; ii++) out_relax << latangle[ii] << "  ";
  out_relax << "\n";
  out_relax << "\n";

  out_relax << "STRAIN MATRIX \n";
  for(int ii=0; ii<3; ii++){
    for(int jj=0; jj<3; jj++){
      out_relax.precision(8);out_relax.width(15);out_relax.setf(ios::showpoint);
      out_relax << strain[ii][jj] << "  ";
    }
    out_relax << "\n";
  }
  out_relax << "\n";

  out_relax << " Atomic relaxations (in the reference system of the relaxed unit cell) \n";
  int count=0;
  for(int na=0; na<atom.size(); na++){
    if(!(atom[na].occ.name[0] == 'V' && atom[na].occ.name[1] == 'a')){
      count++;
      out_relax << "Atom " << count << " relaxation distance = " << atom[na].delta << " Angstrom\n";
    }
  }

  out_relax.close();


  //now place the relaxed coordinates in the structure and set dfcoord=0, dccoord=0, and delta=0.

  for(int na=0; na<atom.size(); na++){
    for(int j=0; j<3; j++){
      atom[na].fcoord[j]=contcar.atom[na].fcoord[j];
      atom[na].ccoord[j]=contcar.atom[na].ccoord[j];
      atom[na].dfcoord[j]=0.0;
      atom[na].dccoord[j]=0.0;
    }
    atom[na].delta=0.0;
  }


  return;

}


//************************************************************

void structure::update_struc(){
  update_lat();
  calc_fractional();
  calc_cartesian();
  calc_factor_group();
  calc_recip_lat();
  get_recip_latparam();

  //test print out lattice parameters and angles
  cout << "lattice parameters a b c \n";
  for(int i=0; i<3; i++)cout << latparam[i] << " ";
  cout << "\n";
  cout << "lattice angle alpha beta gamma\n";
  for(int i=0; i<3; i++)cout << latangle[i] << " ";
  cout << "\n";
  cout << "lattice parameters in order of descending length \n";
  for(int i=0; i<3; i++)cout << latparam[permut[i]] << " ";
  cout << "\n";

}


//************************************************************

void structure::get_recip_latparam(){

  latticeparam(recip_lat, recip_latparam, recip_latangle, recip_permut);

}



//************************************************************

void concentration::collect_components(structure &prim){

  if(prim.atom.size() == 0){
    cout << "cannot collect_components since no atoms in structure \n";
    return;
  }

  for(int i=0; i<prim.atom.size(); i++){
    prim.atom[i].assign_spin();
  }

  //find the first atom that has a min_num_components=2
  int i=0;
  while(prim.atom[i].compon.size() < 2 && i<prim.atom.size()){
    i++;
  }
  if(i==prim.atom.size()){
    cout << "no atoms with more than or equal to 2 components\n";
    cout << "not collecting components\n";
    return;
  }

  compon.clear();
  compon.push_back(prim.atom[i].compon);

  //fill the occup vector with zeros
  occup.clear();
  mu.clear();
  vector<double> toccup;
  for(int k=0; k<prim.atom[i].compon.size(); k++){
    toccup.push_back(0);
  }
  occup.push_back(toccup);
  mu.push_back(toccup);

  for(int j=i+1; j<prim.atom.size(); j++){
    if(prim.atom[j].compon.size() >= 2){
      int k=0;
      for(k=0; k<compon.size(); k++)
	if(compare(prim.atom[j].compon,compon[k])) break;
      if(k==compon.size()){
	compon.push_back(prim.atom[j].compon);
	vector<double> toccup;
	for(int l=0; l<prim.atom[j].compon.size(); l++){
	  toccup.push_back(0);
	}
	occup.push_back(toccup);
	mu.push_back(toccup);
      }
    }
  }

  return;
}



//************************************************************

void concentration::calc_concentration(structure &struc){

  occup.clear();
  for(int i=0; i<compon.size(); i++){
    vector<double> toccup;
    double total=0.0;
    double correction=0.0;
    for(int j=0; j<compon[i].size(); j++){
      double conc=0.0;
      for(int k=0; k<struc.atom.size(); k++){
	//modified by anton
	if(compare(struc.atom[k].compon,compon[i]) && compare(struc.atom[k].occ,compon[i][j])){
	  if(struc.atom[k].basis_flag != '1'){
	    conc=conc+1.0;
	    total=total+1.0;
	  }
	  else{ //Uncomment followin lines to neglect vacancies when calculating concentrations in the occupation basis
	    //if(!(compon[i][j].name[0] == 'V' && compon[i][j].name[1] == 'a')){ //Commented by John.
	    conc=conc+1.0;
	    total=total+1.0; //Added by John
	    //  correction=correction-1.0; //Commented by John.
	    //} //Commented by John.
	  }
	}
      }
      toccup.push_back(conc);
    }
    if(total > tol)
      for(int j=0; j<compon[i].size(); j++){
	//modified by anton - not fool proof
	if(compon[i][j].name[0] == 'V' && compon[i][j].name[1] == 'a') toccup[j]=toccup[j]+correction;
	toccup[j]=toccup[j]/total;
      }
    occup.push_back(toccup);
  }
}



//************************************************************

void concentration::print_concentration(ostream &stream){
  for(int i=0; i<compon.size(); i++){
    for(int j=0; j<compon[i].size()-1; j++){
      compon[i][j].print(stream);
      stream << "=" << occup[i][j] << "  ";
    }
  }
}


//************************************************************

void concentration::print_concentration_without_names(ostream &stream){
  for(int i=0; i<compon.size(); i++){
    for(int j=0; j<compon[i].size()-1; j++){
      stream << occup[i][j] << "  ";
    }
  }
}

//************************************************************

void concentration::print_names(ostream &stream){
  for(int i=0; i<compon.size(); i++){
    for(int j=0; j<compon[i].size()-1; j++){
      compon[i][j].print(stream);
      stream << "  ";
    }
  }
}

//************************************************************

void concentration::get_occup(istream &stream){  // added by jishnu
  for(int i=0; i<compon.size(); i++){
    double sum=0.0;
    for(int j=0; j<compon[i].size()-1; j++){
      stream >> occup[i][j];
      sum=sum+occup[i][j];
    }
    occup[i][compon[i].size()-1]=1.0-sum;
  }
}

//************************************************************

void arrangement::get_bit(istream &stream){  // added by jishnu
  int bit1;
  char ch;
  while(!(stream.peek()=='\n')){
    stream.get(ch);
    if(ch==' ') bit1=110;
    if(ch=='0') bit1=0;
    if(ch=='1') bit1=1;
    if(ch=='2') bit1=2;
    if(ch=='3') bit1=3;
    if(ch=='4') bit1=4;
    if(ch=='5') bit1=5;
    if(ch=='6') bit1=6;
    if(ch=='7') bit1=7;
    if(ch=='8') bit1=8;
    if(ch=='9') bit1=9;
    if(bit1!=110){
      bit.push_back(bit1);
    }
  };

}

//************************************************************
//************************************************************

void arrangement::update_ce(){  // added by jishnu
  if (ce==1) return;
  fenergy = cefenergy;
  assemble_coordinate_fenergy();
  ce = 1;
  fp = 0;
  te = 0;
  return;
}
//************************************************************

void arrangement::update_fp(){  // added by jishnu
  if (fp==1) return;
  fenergy = fpfenergy;
  assemble_coordinate_fenergy();
  ce = 0;
  fp = 1;
  te = 0;
  return;
}//************************************************************

void arrangement::update_te(){  // added by jishnu
  if (te==1) return;
  fenergy = energy;
  assemble_coordinate_fenergy();
  ce = 0;
  fp = 0;
  te = 1;
  return;
}
//************************************************************

void concentration::set_zero(){
  for(int i=0; i<occup.size(); i++){
    for(int j=0; j<occup[i].size(); j++){
      occup[i][j]=0.0;
    }
  }
}

//************************************************************

void concentration::normalize(int n){
  for(int i=0; i<occup.size(); i++){
    for(int j=0; j<occup[i].size(); j++){
      occup[i][j]=occup[i][j]/n;
    }
  }

}




//************************************************************

void concentration::increment(concentration conc){
  if(conc.occup.size() != occup.size()){
    cout << "in increment, concentrations are not compatible\n";
    return;
  }
  for(int i=0; i<conc.occup.size(); i++){
    if(conc.occup[i].size() != occup[i].size()){
      cout << "in increment, concentrations are not compatible\n";
    }
    for(int j=0; j<occup[i].size(); j++){
      occup[i][j]=occup[i][j]+conc.occup[i][j];
    }
  }
}




//************************************************************


void cluster::get_dimensions(){
  double dist,diff;
  min_leng=1.0e20;
  max_leng=0.0;
  for(int i=0; i<point.size(); i++)
    for(int j=0; j<point.size(); j++)
      if(i != j){
	dist=0;
	for(int k=0; k<3; k++){
	  diff=(point[i].ccoord[k]-point[j].ccoord[k]);
	  dist=dist+diff*diff;
	}
	dist=sqrt(dist);
	if(dist < min_leng)min_leng=dist;
	if(dist > max_leng)max_leng=dist;
      }
}


//************************************************************

cluster cluster::apply_sym(sym_op op){
  cluster tclust;
  tclust.min_leng=min_leng;
  tclust.max_leng=max_leng;
  for(int i=0; i<point.size(); i++){
    atompos tatom;
    tatom=point[i].apply_sym(op);
    tclust.point.push_back(tatom);
  }
  tclust.clust_group.clear();
  return tclust;
}


//************************************************************

void cluster::get_cart(double FtoC[3][3]){
  int np;
  for(np=0; np<point.size(); np++)
    conv_AtoB(FtoC,point[np].fcoord,point[np].ccoord);
}


//************************************************************

void cluster::get_frac(double CtoF[3][3]){
  int np;
  for(np=0; np<point.size(); np++)
    conv_AtoB(CtoF,point[np].ccoord,point[np].fcoord);
}


//************************************************************

void cluster::readf(istream &stream, int np){
  //clear out the points before reading new ones
  point.clear();
  for(int i=0; i<np; i++){
    atompos tatom;
    tatom.readf(stream);
    point.push_back(tatom);
  }

}


//************************************************************

void cluster::readc(istream &stream, int np){
  //clear out the points before reading new ones
  point.clear();
  for(int i=0; i<np; i++){
    atompos tatom;
    tatom.readc(stream);
    point.push_back(tatom);
  }

}


//************************************************************

void cluster::print(ostream &stream){
  for(int i=0; i<point.size(); i++){
    point[i].print(stream);
  }
}


//************************************************************

void cluster::write_clust_group(ostream &stream){
  int cg;

  stream << "cluster group for cluster \n";
  print(stream);
  stream << "\n";
  stream << " number of cluster group ops " << clust_group.size() << "\n";

  for(cg=0; cg<clust_group.size(); cg++){
    stream << "cluster group operation " << cg << " \n";
    clust_group[cg].print_fsym_mat(stream);
    stream << "\n";
  }
}


//************************************************************

void cluster::determine_site_attributes(structure prim){
  for(int i=0; i<point.size(); i++){
    //determine which prim site this point maps onto
    //when there is a match - copy all the attributes from prim onto that site

    for(int j=0; j < prim.atom.size(); j++){
      int trans[3];
      if(compare(point[i],prim.atom[j],trans)){
	//copy attributes from prim.atom[j] onto point[i]
	point[i]=prim.atom[j];
	for(int k=0; k<3; k++) point[i].fcoord[k]=point[i].fcoord[k]+trans[k];
      }
    }

  }

}





//************************************************************

void orbit::readf(istream &stream, int np, int mult){
  //clear out the equivalent clusters before reading new ones
  equiv.clear();
  for(int nm=0; nm<mult; nm++){
    char buff[200];
    stream.getline(buff,199);
    cluster tclust;
    tclust.readf(stream,np);
    equiv.push_back(tclust);
  }

}


//************************************************************

void orbit::readc(istream &stream, int np, int mult){
  //clear out the equivalent clusters before reading new ones
  equiv.clear();
  for(int nm=0; nm<mult; nm++){
    char buff[200];
    stream.getline(buff,199);
    cluster tclust;
    tclust.readc(stream,np);
    equiv.push_back(tclust);
  }


}


//************************************************************

void orbit::print(ostream &stream){
  for(int i=0; i<equiv.size(); i++){
    stream << "equivalent cluster " << i+1 << "\n";
    equiv[i].print(stream);
  }
}


//************************************************************

void orbit::get_cart(double FtoC[3][3]){
  for(int ne=0; ne < equiv.size(); ne++){
    equiv[ne].get_cart(FtoC);
    equiv[ne].get_dimensions();
  }
  return;
}



//************************************************************

void orbit::get_frac(double CtoF[3][3]){
  for(int ne=0; ne < equiv.size(); ne++){
    equiv[ne].get_frac(CtoF);
  }
  return;
}


//************************************************************

void orbit::determine_site_attributes(structure prim){
  for(int ne=0; ne < equiv.size(); ne++){
    equiv[ne].determine_site_attributes(prim);
  }
}



//************************************************************

void multiplet::readf(istream &stream){
  char buff[200];
  char bull;

  //clear out the orbit before reading clusters
  orb.clear();
  size.clear();
  order.clear();
  index.clear();

  vector<orbit> torbvec;

  //make the empty cluster and put it in orb
  {
    cluster tclust;
    tclust.max_leng=0;
    tclust.min_leng=0;
    orbit torb;
    torb.equiv.push_back(tclust);
    torbvec.push_back(torb);
    //the first index i.e. 0 is always for the empty cluster
    //    int i=0;
    size.push_back(0);
    order.push_back(0);
  }


  //read the orbits and collect them torbvec

  int max_np=0;
  int nc;
  stream >> nc;
  stream.get(bull);
  for(int i=1; i<= nc; i++){
    stream.getline(buff,199);

    int dummy,np,mult;
    stream >> dummy;
    stream >> np;
    stream >> mult;
    stream >> dummy;
    stream.getline(buff,199);
    if(np > max_np) max_np=np;
    orbit torb;
    torb.readf(stream,np,mult);
    size.push_back(np);
    order.push_back(0);
    torbvec.push_back(torb);
  }



  //then for all cluster sizes less than or equal to max_np, collect all orbits of the same size
  //we also keep track of the indexing so we remember the order in which the clusters were input
  //(necessary for eci match up)

  for(int np=0; np<=max_np; np++){

    vector<orbit> orbvec;
    vector<int> tindex;
    for(int i=0; i<=nc; i++){
      if(size[i] == np){
	orbvec.push_back(torbvec[i]);
	tindex.push_back(i);
	order[i]=tindex.size()-1;
      }
    }
    orb.push_back(orbvec);
    index.push_back(tindex);
  }

}



//************************************************************

void multiplet::readc(istream &stream){

}



//************************************************************

void multiplet::print(ostream &stream){
  for(int i=0; i<orb.size(); i++)
    for(int j=0; j<orb[i].size(); j++)
      orb[i][j].print(stream);
}


//************************************************************

void multiplet::sort(int np){

  for(int i=0; i<orb[np].size(); i++){
    for(int j=i+1; j<orb[np].size(); j++){
      if(orb[np][i].equiv[0].max_leng > orb[np][j].equiv[0].max_leng){
	orbit torb=orb[np][j];
	orb[np][j]=orb[np][i];
	orb[np][i]=torb;
      }
    }
  }
}


//************************************************************

void multiplet::get_index(){
  int count=0;

  size.clear();
  order.clear();
  index.clear();

  //first the emtpy cluster
  size.push_back(0);
  order.push_back(0);
  vector <int> tindex;
  tindex.push_back(count);
  index.push_back(tindex);

  for(int np=1; np<orb.size(); np++){
    vector <int> tindex;
    for(int nc=0; nc<orb[np].size(); nc++){
      count++;
      tindex.push_back(count);
      size.push_back(np);
      order.push_back(nc);
    }
    index.push_back(tindex);
  }

  return;

}


//************************************************************

void multiplet::get_hierarchy(){

  int count=0;

  size.clear();
  order.clear();
  index.clear();
  subcluster.clear();

  //first the emtpy cluster
  size.push_back(0);
  order.push_back(0);
  vector <int> tindex;
  tindex.push_back(count);
  index.push_back(tindex);

  for(int np=1; np<orb.size(); np++){
    vector <int> tindex;
    for(int nc=0; nc<orb[np].size(); nc++){
      count++;
      tindex.push_back(count);
      size.push_back(np);
      order.push_back(nc);
    }
    index.push_back(tindex);
  }


  // make the subcluster table for the empty cluster
  {
    vector<int>temptysubcluster;
    subcluster.push_back(temptysubcluster);
  }

  // make theh subcluster tables for the non-empty clusters
  for(int np=1; np<orb.size(); np++){

    for(int nc=0; nc<orb[np].size(); nc++){
      vector<int> tsubcluster;

      //enumerate all subclusters of the cluster orbit with index (np,nc)
      //find which cluster this subcluster is equivalent to
      //record the result in tsubcluster

      for(int snp=1; snp<np; snp++){
	vector<int> sc;            // contains the indices of the subcluster
	for(int i=0; i<snp; i++) sc.push_back(i);

	while(sc[0]<=(np-snp)){
	  while(sc[snp-1]<=(np-1)){

	    //BLOCK WHERE SUB CLUSTER IS FOUND AMONG THE LIST

	    cluster tclust;
	    for(int i=0; i<snp; i++)
	      tclust.point.push_back(orb[np][nc].equiv[0].point[sc[i]]);
	    within(tclust);



	    //compare the subclusters with all clusters of the same size

	    for(int i=0; i<orb[snp].size(); i++){
	      if(!new_clust(tclust,orb[snp][i])){
		// snp,i is a subcluster
		// check whether it already exists among the subcluster list

		int j;
		for(j=0; j<tsubcluster.size(); j++){
		  if(index[snp][i] == tsubcluster[j])break;
		}
		if(j== tsubcluster.size()) tsubcluster.push_back(index[snp][i]);

	      }

	    }

	    //END OF BLOCK TO DETERMINE SUBCLUSTERS

	    sc[snp-1]++;
	  }
	  int j=snp-2;
	  if(j>-1){
	    while(sc[j] == (np-(snp-j)) && j>-1) j--;

	    if(j>-1){
	      sc[j]++;
	      for(int k=j+1; k<snp; k++) sc[k]=sc[k-1]+1;
	    }
	    else break;

	  }
	}
      }
      // extra for pair clusters (and possibly point clusters)

      if(np == 2){
	if(nc>0){
	  int k=nc-1;
	  while(k>=0 && abs(orb[np][k].equiv[0].max_leng-orb[np][nc].equiv[0].max_leng) < 0.0001)k--;
	  if(k>=0 && orb[np][nc].equiv[0].max_leng-orb[np][k].equiv[0].max_leng >= 0.0001){
	    tsubcluster.push_back(index[np][k]);
	  }
	}

      }

      subcluster.push_back(tsubcluster);
    }
  }

}


//************************************************************

void multiplet::print_hierarchy(ostream &stream){
  stream << "label    weight    mult    size    length    heirarchy \n";
  for(int i=0; i<subcluster.size(); i++){
    stream << i << "   " << "   0    " << orb[size[i]][order[i]].equiv.size() << "   ";
    stream << orb[size[i]][order[i]].equiv[0].point.size() << "   ";
    stream << orb[size[i]][order[i]].equiv[0].max_leng << "   ";
    stream << subcluster[i].size() << "   ";
    for(int j=0; j<subcluster[i].size(); j++) stream << subcluster[i][j] << "   ";
    stream << "\n";
  }

}


//************************************************************

void multiplet::read_eci(istream &stream){
  char buff[200];
  double eci1,eci2;
  int index;
  get_hierarchy();
  for(int i=0; i<7; i++)stream.getline(buff,199);
  while(!stream.eof()){
    stream >> eci1;
    stream >> eci2;
    stream >> index;
    if(index >=size.size()){
      cout << "WARNING:  eci.out has cluster indeces larger than total number of clusters.  Please check for source of incompatibility.  Exiting...\n";
      exit(1);
    }
    stream.getline(buff,199);
    //cout << eci1 << "  " << eci2 << "  " << index << "\n";
    orb[size[index]][order[index]].eci=eci2;
  }

}


//************************************************************

void multiplet::get_cart(double FtoC[3][3]){
  for(int np=1; np < orb.size(); np++){
    for(int ne=0; ne < orb[np].size(); ne++){
      orb[np][ne].get_cart(FtoC);
    }
  }
  return;
}





//************************************************************

void multiplet::get_frac(double CtoF[3][3]){
  for(int np=1; np < orb.size(); np++){
    for(int ne=0; ne < orb[np].size(); ne++){
      orb[np][ne].get_frac(CtoF);
    }
  }
  return;
}



//************************************************************

void multiplet::determine_site_attributes(structure prim){
  for(int np=0; np<orb.size(); np++){
    for(int no=0; no<orb[np].size(); no++){
      orb[np][no].determine_site_attributes(prim);
    }
  }

}


//************************************************************

void arrangement::assemble_coordinate_fenergy(){
  coordinate.clear();
  for(int i=0; i<conc.occup.size(); i++){
    for(int j=0; j<conc.occup[i].size()-1; j++){
      coordinate.push_back(conc.occup[i][j]);
    }
  }
  coordinate.push_back(fenergy);

}


//************************************************************
//************************************************************
/*
  void arrangement::assemble_coordinate_CE(){
  coordinate_CE.clear();
  for(int i=0; i<conc.occup.size(); i++){
  for(int j=0; j<conc.occup[i].size()-1; j++){
  coordinate_CE.push_back(conc.occup[i][j]);
  }
  }
  coordinate_CE.push_back(cefenergy);

  }*/


//************************************************************
void arrangement::print_bit(ostream &stream){
  for(int i=0; i<bit.size(); i++) stream << bit[i] << " ";
  stream << "\n";
}



//************************************************************

void arrangement::print_correlations(ostream &stream){
  for(int i=0; i<correlations.size(); i++){
    stream.precision(8);stream.width(12);stream.setf(ios::showpoint);
    stream << correlations[i] << " ";
  }
  stream << "\n";

}


//************************************************************

void arrangement::print_coordinate(ostream &stream){


  for(int i=0; i<coordinate.size(); i++){
    stream.precision(8);stream.width(12);stream.setf(ios::showpoint);
    stream << coordinate[i] << " ";
  }
  stream << name;
  stream << "\n";

}


//************************************************************
//************************************************************

void arrangement::print_in_energy_file(ostream &stream){    //added by jishnu
  stream.precision(8);stream.width(12);stream.setf(ios::showpoint);
  stream << fenergy << "   ";
  stream.precision(8);stream.width(12);stream.setf(ios::showpoint);
  stream << weight << "   ";
  for(int i=0; i<coordinate.size()-1; i++){
    stream.precision(8);stream.width(12);stream.setf(ios::showpoint);
    stream << coordinate[i] << " ";
  }
  stream.precision(8);stream.width(12);stream.setf(ios::showpoint);
  stream << delE_from_facet << "   ";
  stream.precision(8);stream.width(12);stream.setf(ios::showpoint);
  stream << name;
  stream << "\n";

}


//************************************************************

void superstructure::decorate_superstructure(arrangement conf){

  if(struc.atom.size() != conf.bit.size()){
    cout << "inside decorate_superstructure and conf.bit.size() is \n";
    cout << "not equal to the number of atoms in the structure \n";
    exit(1);
  }


  for(int i=0; i<struc.atom.size(); i++){
    struc.atom[i].occ=struc.atom[i].compon[conf.bit[i]];
  }


  //collect the components within the structure
  struc.collect_components();
  curr_struc=struc;
  curr_struc.num_each_specie.clear();


  int curr_ind=0;
  for(int nc=0; nc<struc.compon.size(); nc++){
    int num=0;
    for(int i=0; i<struc.atom.size(); i++){
      if(compare(struc.compon[nc],struc.atom[i].occ)){
	curr_struc.atom[curr_ind]=struc.atom[i];
	curr_ind++;
	num++;
      }
    }
    curr_struc.num_each_specie.push_back(num);
  }


}


//************************************************************

void superstructure::determine_kpoint_grid(double kpoint_dens){

  struc.calc_recip_lat();
  struc.get_recip_latparam();
  double recip_vol=determinant(struc.recip_lat);

  double mleng=struc.recip_latparam[struc.recip_permut[0]];

  int i=0;
  do{
    i++;
    double delta=mleng/i;
    for(int j=0; j<3; j++)
      kmesh[j]=int(ceil(struc.recip_latparam[j]/delta));
  }while((kmesh[0]*kmesh[1]*kmesh[2])/recip_vol < kpoint_dens && i < 99);

  if(i > 99){
    cout << "k-point grid is unusually high \n";
  }


}


//************************************************************

void superstructure::print_incar(string dir_name){


  if(!scandirectory(dir_name,"INCAR")){

    string outfile = dir_name;
    outfile.append("/INCAR");

    string infile = "INCAR";

    ifstream in;
    in.open(infile.c_str());

    if(!in){
      cout << "cannot open parent INCAR file \n";
      cout << "no INCAR can be created for the configurations.\n";
      return;
    }

    ofstream out(outfile.c_str());

    if(!out){
      cout << "no INCAR created for the configuration " <<  dir_name <<".\n";
      return;
    }

    string line;
    while(getline(in,line)){
      // adding magnetic moments
      bool spin_pol=false;
      string check=line.substr(0,5);
      string check2=line.substr(0,6);
      if (  (check == "ISPIN") && (check2 != "ISPIND") ){
	for(int i=0;i<line.size();i++){
	  if(line[i]=='1') {spin_pol=false;break;}
	  if(line[i]=='2') {spin_pol=true;break;}
	}
      }
      if(spin_pol) {
	out << "MAGMOM = ";
	for(int i=0; i<curr_struc.atom.size(); i++){
	  if(!(curr_struc.atom[i].occ.name[0] == 'V' && curr_struc.atom[i].occ.name[1] == 'a')){
	    out << curr_struc.atom[i].occ.magmom;
	    out << " ";
	  }
	}
	out << "\n";
      }  // end of adding magmoms
      // writing L's
      bool Lline=false;
      check=line.substr(0,5);
      if (check == "LDAUL") {
	Lline = true;
	out << "LDAUL = ";
	for(int i=0; i < curr_struc.num_each_specie.size(); i++){
	  if(!(curr_struc.compon[i].name[0] == 'V' && curr_struc.compon[i].name[1] == 'a') && curr_struc.num_each_specie[i] != 0)
	    out << "  2";
	}
	out << "\n";
      }  // end of writing L's
      // writing U's
      bool Uline=false;
      check=line.substr(0,5);
      if (check == "LDAUU") {
	Uline = true;
	out << "LDAUU = ";
	for(int i=0; i < curr_struc.num_each_specie.size(); i++){
	  if(!(curr_struc.compon[i].name[0] == 'V' && curr_struc.compon[i].name[1] == 'a') && curr_struc.num_each_specie[i] != 0)
	    out << " " << curr_struc.compon[i].U;
	}
	out << "\n";
      }  // end of writing U's
      // writing J's
      bool Jline=false;
      check=line.substr(0,5);
      if (check == "LDAUJ") {
	Jline = true;
	out << "LDAUJ = ";
	for(int i=0; i < curr_struc.num_each_specie.size(); i++){
	  if(!(curr_struc.compon[i].name[0] == 'V' && curr_struc.compon[i].name[1] == 'a') && curr_struc.num_each_specie[i] != 0)
	    out << " " << curr_struc.compon[i].J;
	}
	out << "\n";
      }  // end of writing J's

      if(!Lline && !Uline && !Jline) {
	out << line;
	out << "\n";
      }


    }

    in.close();
    out.close();

    return;
  }
  return;

}


//************************************************************
//************************************************************

void superstructure::print(string dir_name, string file_name){

  if(!scandirectory(dir_name,file_name)){

    string file=dir_name;
    file.append("/");
    file.append(file_name);

    ofstream out(file.c_str());

    if(!out){
      cout << "cannot open " << file << "\n";
      return;
    }

    out << curr_struc.title <<"\n";

    out.precision(7);out.width(12);out.setf(ios::showpoint);
    out << curr_struc.scale <<"\n";

    for(int i=0; i<3; i++){
      out << "  ";
      for(int j=0; j<3; j++){

	out.precision(9);out.width(15);out.setf(ios::showpoint);
	out << curr_struc.lat[i][j] << " ";

      }
      out << "\n";
    }
    for(int i=0; i < curr_struc.num_each_specie.size(); i++){
      if(!(curr_struc.compon[i].name[0] == 'V' && curr_struc.compon[i].name[1] == 'a') && curr_struc.num_each_specie[i] != 0)
	out << " " << curr_struc.num_each_specie[i] ;
    }
    out << "\n";

    out << "Direct\n";

    for(int i=0; i<curr_struc.atom.size(); i++){
      if(!(curr_struc.atom[i].occ.name[0] == 'V' && curr_struc.atom[i].occ.name[1] == 'a')){
	for(int j=0; j<3; j++){
	  out.precision(9);out.width(15);out.setf(ios::showpoint);
	  out << curr_struc.atom[i].fcoord[j] << " ";
	}
	out << curr_struc.atom[i].occ.name;  // jishnu
	out << "\n";
      }
    }
    out.close();
    return;
  }

}


//************************************************************

void superstructure::print_potcar(string dir_name){

  string potcar_file=dir_name;
  potcar_file.append("/POTCAR");

  //  ofstream out;
  //  out.open(potcar_file.c_str());
  //  if(!out){
  //    cout << "cannot open POTCAR file.\n";
  //    return;
  //  }
  //  out.close();

  //look at the first atom
  string last_element;

  string command = "cat ";

  if(!(curr_struc.atom[0].occ.name[0] == 'V' && curr_struc.atom[0].occ.name[1] == 'a')){
    string element=curr_struc.atom[0].occ.name;  // jishnu
    string potcar="POTCAR_";
    potcar.append(element);
    command.append(potcar);

    last_element=element;
  }

  for(int i=1; i<curr_struc.atom.size(); i++){
    if(!(curr_struc.atom[i].occ.name[0] == 'V' && curr_struc.atom[i].occ.name[1] == 'a')){
      string element=curr_struc.atom[i].occ.name;  // jishnu
      if(element != last_element){
	string potcar="POTCAR_";
	potcar.append(element);
	command.append(" ");
	command.append(potcar);

	last_element=element;
      }

    }
  }

  command.append(" > ");
  command.append(potcar_file);
  //    cout << command << "\n";

  int s=system(command.c_str());
  if(s == -1){cout << "was unable to perform system command\n";}


  return;
}





//************************************************************

void superstructure::print_kpoint(string dir_name){


  if(!scandirectory(dir_name,"KPOINTS")){

    string file_name=dir_name;
    file_name.append("/KPOINTS");

    ofstream out;
    out.open(file_name.c_str());
    if(!out){
      cout << "cannot open " << file_name << "\n";
      return;
    }
    out << dir_name << "\n";
    out << " 0 \n";
    out << "Gamma point shift\n";
    for(int j=0; j<3; j++)out << " " << kmesh[j];
    out << "\n";
    out << " 0 0 0 \n";
    out.close();
  }

}


//************************************************************
//************************************************************
//read the configuration folder names to check whether a particular one is already present   //added by jishnu
bool read_vasp_list_file(string name) {

  string s;
  ifstream rf;
  rf.open("vasp_list_file",ios::out);
  do{
    rf>>s;
    if (s==name) {rf.close();return true;}
  }while (!rf.eof());
  return false;
}
//************************************************************
//************************************************************
//write the configuration folder names to a file so that automatic submitvasp can work   //added by jishnu
void write_vasp_list_file(string name) {

  bool ifthere;
  if(!scandirectory(".","vasp_list_file")){
    string file_name="vasp_list_file";
    ofstream out;
    out.open(file_name.c_str());
    if(!out){
      cout << "cannot open " << file_name << "\n";
      return;
    }
    out.close();
  }

  ifthere=read_vasp_list_file(name);

  if(!ifthere) {
    ofstream out;
    out.open("vasp_list_file",ios::app);
    out << "D "<< name <<"\n";
    out.close();
  }  // end of if(!ifthere)

  return;

}
//************************************************************
//************************************************************
//write the structure to a file in yihaw like format   //added by jishnu

void superstructure::print_yihaw(string dir_name) {


  if(!scandirectory(dir_name,"yihaw")){

    string file_name=dir_name;
    file_name.append("/yihaw");

    ofstream out;
    out.open(file_name.c_str());
    if(!out){
      cout << "cannot open " << file_name << "\n";
      return;
    }

    out << "#!/bin/sh" <<"\n";
    out << "#PBS -S /bin/sh" <<"\n";
    out << "#PBS -N vasp" <<"\n";
    out << "#PBS -l nodes="<< nodes <<":ppn="<< ppn <<",walltime=" << walltime <<":00:00"<<"\n";
    out << "#PBS -q "<< queue <<"\n";
    out << "\n";
    out << "#PBS -o " << parent_directory << "/" <<dir_name<<"\n";
    out << "#PBS -e " << parent_directory << "/" <<dir_name<<"\n";
    out << "#PBS -joe" <<"\n";
    out << "#PBS -V" <<"\n";
    out << "#" <<"\n";
    out << "\n";
    //out << "echo ""<< "l ran on:"<<"""<<"\n";
    out << "cat $PBS_NODEFILE" <<"\n";
    out << "#" <<"\n";
    out << "# Change to your execution directory." <<"\n";
    out << "cd " << parent_directory << "/" <<dir_name<<"\n";
    out << "#" <<"\n";
    out << "\n";
    out << "lamboot" <<"\n";
    out << "\n";
    out << "#" <<"\n";
    out << "# Use mpirun to run with "<< ppn << " cpu for "<< walltime <<" hours" <<"\n";
    out << "\n";
    out << "mpirun  -np "<< ppn << " vasp" <<"\n";
    out << "\n";
    out << "lamhalt" <<"\n";
    out << "#" <<"\n";

    out.close();

  }
}



//************************************************************
//************************************************************
//read the yihaw_input file   //added by jishnu

void superstructure::read_yihaw_input() {

  if(!scandirectory(".","yihaw_input")){
    cout << "No yihaw_input file to open  \n";
  }

  ifstream readfrom;
  int n;
  char ch;
  readfrom.open("yihaw_input",ios::out);
  n=0;
  do{
    readfrom.get(ch);
    if(ch=='=') n=n+1;
  }while(n<1);
  readfrom>>nodes;
  n=0;
  do{
    readfrom.get(ch);
    if(ch=='=') n=n+1;
  }while(n<1);
  readfrom>>ppn;
  n=0;
  do{
    readfrom.get(ch);
    if(ch=='=') n=n+1;
  }while(n<1);
  readfrom>>walltime;
  n=0;
  do{
    readfrom.get(ch);
    if(ch=='=') n=n+1;
  }while(n<1);
  readfrom>>queue;
  n=0;
  do{
    readfrom.get(ch);
    if(ch=='=') n=n+1;
  }while(n<1);
  readfrom>>parent_directory;

  readfrom.close();

}

//************************************************************

void structure::calc_recip_lat(){
  double pi=3.141592654;
  double vol=determinant(lat);

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++) recip_lat[i][j]=2.0*pi*(lat[(i+1)%3][(j+1)%3]*lat[(i+2)%3][(j+2)%3]-
						   lat[(i+1)%3][(j+2)%3]*lat[(i+2)%3][(j+1)%3])/vol;
  }
}



//************************************************************
////////////////////////////////////////////////////////////////////////////////
//added by Ben Swoboda
//takes atompos as input and determines the basis vectors for occupation and spin methods
void get_basis_vectors(atompos &atom){

  int tspin;
  atom.spin_vec.clear();
  atom.p_vec.clear();

  tspin=1;
  //modified by Anton (...compon.size()-1 ...)
  for(int i=0; i < atom.compon.size()-1; i++){
    tspin=tspin*atom.occ.spin;
    atom.spin_vec.push_back(tspin);
  }


  //  atom.p_vec.push_back(1);
  //modified by Anton (i=0 and compon.size()-1 instead of i=1 and compon.size() )
  for(int i=0; i<atom.compon.size()-1; i++){
    if(compare(atom.occ,atom.compon[i]))atom.p_vec.push_back(1);
    else atom.p_vec.push_back(0);
  }

  return;

}
////////////////////////////////////////////////////////////////////////////////
//************************************************************

void configurations::generate_configurations(vector<structure> supercells){
  int ns,i;

  for(ns=0; ns<supercells.size(); ns++){
    superstructure tsuperstruc;
    multiplet super_basiplet;
    //used to be copy_lattice below
    tsuperstruc.struc=supercells[ns];
    tsuperstruc.struc.expand_prim_basis(prim);
    tsuperstruc.struc.expand_prim_clust(basiplet,super_basiplet);

    //generate the different bit combinations

    int last=0;

    while(last == 0){
      tsuperstruc.struc.atom[0].bit++;
      for(i=0; i<(tsuperstruc.struc.atom.size()-1); i++){
	if(tsuperstruc.struc.atom[i].bit !=0 &&
	   tsuperstruc.struc.atom[i].bit%(tsuperstruc.struc.atom[i].compon.size()) == 0){
	  tsuperstruc.struc.atom[i+1].bit++;
	  tsuperstruc.struc.atom[i].bit=0;
	}
      }
      if(tsuperstruc.struc.atom[tsuperstruc.struc.atom.size()-1].bit !=0 &&
	 tsuperstruc.struc.atom[tsuperstruc.struc.atom.size()-1].bit%(tsuperstruc.struc.atom[tsuperstruc.struc.atom.size()-1].compon.size()) == 0){
	last=last+1;
	tsuperstruc.struc.atom[tsuperstruc.struc.atom.size()-1].bit=0;
      }

      //for each atom position, assign its spin and specie for this configuration of bits

      arrangement tconf;

      for(i=0; i<tsuperstruc.struc.atom.size(); i++){
	tsuperstruc.struc.atom[i].occ=tsuperstruc.struc.atom[i].compon[tsuperstruc.struc.atom[i].bit];
	////////////////////////////////////////////////////////////////////////////////
	//modified by Ben Swoboda
	//assign the spin to the atompos object for use in basis
	get_basis_vectors(tsuperstruc.struc.atom[i]);


	//cout  << "p-vec[" << i << "]\n";
	//for(int x=0; x<tsuperstruc.struc.atom[i].p_vec.size(); x++){
	//    cout << tsuperstruc.struc.atom[i].p_vec[x] << "\t";
	//}
	//cout << "\n";
	//cout << "bit[" << i << "]: " << tsuperstruc.struc.atom[i].bit << "\tspin: " << tsuperstruc.struc.atom[i].occ.spin << "\tspecie: "
	//<< tsuperstruc.struc.atom[i].occ.name << "\n";
	////////////////////////////////////////////////////////////////////////////////
	tconf.bit.push_back(tsuperstruc.struc.atom[i].bit);
      }
      calc_correlations(tsuperstruc.struc,super_basiplet,tconf);


      //calculate the concentration over the sites with more than min_num_components = 2
      tconf.conc.collect_components(prim);
      tconf.conc.calc_concentration(tsuperstruc.struc);


      if(new_conf(tconf,superstruc) && new_conf(tconf,tsuperstruc)){

	//give this newly found configuration its name
	tconf.name="con";
	string scel_num;
	string period=".";
	int_to_string(ns,scel_num,10);
	tconf.name.append(scel_num);
	tconf.name.append(period);
	string conf_num;
	int_to_string(tsuperstruc.conf.size(),conf_num,10);
	tconf.name.append(conf_num);

	//record the indices of this configuration
	tconf.ns=ns;
	tconf.nc=tsuperstruc.conf.size();

	//add the new configuration to the list for this superstructure
	tsuperstruc.conf.push_back(tconf);

      }

    }
    superstruc.push_back(tsuperstruc);
  }

  return;
}

//************************************************************

void configurations::generate_configurations_fast(vector<structure> supercells){
  int ns,i,j,k;
  double tcorr, tclust_func;

  for(ns=0; ns<supercells.size(); ns++){
    superstructure tsuperstruc;
    multiplet super_basiplet;
    //used to be copy_lattice below
    tsuperstruc.struc=supercells[ns];
    tsuperstruc.struc.expand_prim_basis(prim);
    tsuperstruc.struc.expand_prim_clust(basiplet,super_basiplet);

    //get cluster function and basis info for this supercell
    get_corr_vector(tsuperstruc.struc, super_basiplet, tsuperstruc.corr_to_atom_vec);
    get_super_basis_vec(tsuperstruc.struc, tsuperstruc.basis_to_bit_vec);

    //generate the different bit combinations

    int last=0;

    while(last == 0){
      tsuperstruc.struc.atom[0].bit++;
      for(i=0; i<(tsuperstruc.struc.atom.size()-1); i++){
	if(tsuperstruc.struc.atom[i].bit !=0 &&
	   tsuperstruc.struc.atom[i].bit%(tsuperstruc.struc.atom[i].compon.size()) == 0){
	  tsuperstruc.struc.atom[i+1].bit++;
	  tsuperstruc.struc.atom[i].bit=0;
	}
      }
      if(tsuperstruc.struc.atom[tsuperstruc.struc.atom.size()-1].bit !=0 &&
	 tsuperstruc.struc.atom[tsuperstruc.struc.atom.size()-1].bit%(tsuperstruc.struc.atom[tsuperstruc.struc.atom.size()-1].compon.size()) == 0){
	last=last+1;
	tsuperstruc.struc.atom[tsuperstruc.struc.atom.size()-1].bit=0;
      }

      //for each atom position, assign its spin and specie for this configuration of bits

      arrangement tconf;

      for(i=0; i<tsuperstruc.struc.atom.size(); i++){
	tsuperstruc.struc.atom[i].occ=tsuperstruc.struc.atom[i].compon[tsuperstruc.struc.atom[i].bit];
	tconf.bit.push_back(tsuperstruc.struc.atom[i].bit);
      }
      tconf.correlations.push_back(1.0);
      for(i=0; i<tsuperstruc.corr_to_atom_vec.size(); i++){
	tcorr=0.0;
	for(j=0; j<tsuperstruc.corr_to_atom_vec[i].size(); j++){
	  tclust_func=1.0;
	  for(k=0; k<tsuperstruc.corr_to_atom_vec[i][j].size(); k++){
	    tclust_func*=tsuperstruc.basis_to_bit_vec[tsuperstruc.corr_to_atom_vec[i][j][k][0]][tconf.bit[tsuperstruc.corr_to_atom_vec[i][j][k][0]]][tsuperstruc.corr_to_atom_vec[i][j][k][1]];
	  }
	  tcorr+=tclust_func;
	}
	tconf.correlations.push_back(tcorr/tsuperstruc.corr_to_atom_vec[i].size());
      }




      if(new_conf(tconf,superstruc) && new_conf(tconf,tsuperstruc)){

	//calculate the concentration over the sites with more than min_num_components = 2
	tconf.conc.collect_components(prim);
	tconf.conc.calc_concentration(tsuperstruc.struc);

	//give this newly found configuration its name
	tconf.name="con";
	string scel_num;
	string period=".";
	int_to_string(ns,scel_num,10);
	tconf.name.append(scel_num);
	tconf.name.append(period);
	string conf_num;
	int_to_string(tsuperstruc.conf.size(),conf_num,10);
	tconf.name.append(conf_num);

	//record the indices of this configuration
	tconf.ns=ns;
	tconf.nc=tsuperstruc.conf.size();

	//add the new configuration to the list for this superstructure
	tsuperstruc.conf.push_back(tconf);

      }

    }
    superstruc.push_back(tsuperstruc);
  }

  return;
}



//************************************************************

void configurations::generate_vasp_input_directories(){

  //first determine the kpoints density

  double kpt_dens;
  {
    int kmesh[3];

    // read in the kpoints for the primitive cell
    string kpoint_file="KPOINTS";
    ifstream in_kpoints;
    in_kpoints.open(kpoint_file.c_str());
    if(!in_kpoints){
      cout << "cannot open " << kpoint_file << "\n";
      return;
    }
    char buff[200];
    for(int i=0; i<3; i++) in_kpoints.getline(buff,199);
    for(int j=0; j<3; j++)in_kpoints >> kmesh[j];
    in_kpoints.close();

    double recip_vol=determinant(prim.recip_lat);
    kpt_dens=(kmesh[0]*kmesh[1]*kmesh[2])/recip_vol;
  }


  //create vasp files for all the configurations

  for(int sc=0; sc<superstruc.size(); sc++){

    //make the kpoint-mesh for this superstructure

    superstruc[sc].determine_kpoint_grid(kpt_dens);

    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){

      if(!scandirectory(".",superstruc[sc].conf[nc].name) && superstruc[sc].conf[nc].make){
        write_vasp_list_file(superstruc[sc].conf[nc].name);  // added by jishnu for auto submission of vasp runs
        string command="mkdir ";
        command.append(superstruc[sc].conf[nc].name);
        int s=system(command.c_str());
        if(s == -1){cout << "was unable to perform system command\n";}
      }

      if(scandirectory(".",superstruc[sc].conf[nc].name)){

	// string command="cp INCAR ";    // This part is from the time when we just copied INCAR not write it
	// command.append(superstruc[sc].conf[nc].name);
	// int s=system(command.c_str());
	// if(s == -1){cout << "was unable to perform system command\n";}

	superstruc[sc].decorate_superstructure(superstruc[sc].conf[nc]);

	superstruc[sc].print(superstruc[sc].conf[nc].name,"POSCAR");

	superstruc[sc].print(superstruc[sc].conf[nc].name,"POS");

	superstruc[sc].print_incar(superstruc[sc].conf[nc].name);      // added by jishnu  // this is to explicitly write the INCAR based upon the parent INCAR in parent directory

	superstruc[sc].print_potcar(superstruc[sc].conf[nc].name);

	superstruc[sc].print_kpoint(superstruc[sc].conf[nc].name);

	//superstruc[sc].read_yihaw_input();   // added by jishnu

	//superstruc[sc].print_yihaw(superstruc[sc].conf[nc].name);   // added by jishnu
      }
    }
  }

}


//************************************************************

void configurations::print_con_old(){
  int i,j,k;

  ofstream out;
  out.open("CON");
  if(!out){
    cout << "cannot open CON \n";
    return;
  }

  out << "Structures generated within supercells \n";
  out << "\n";
  out << superstruc.size() << "  supercells considered\n";
  for(int sc=0; sc<superstruc.size(); sc++){
    out << "\n";
    for(i=0; i<3; i++){
      for(j=0; j<3; j++){
	out.precision(5);out.width(5);
	out << superstruc[sc].struc.slat[i][j] << " ";
      }
      out << "  ";
    }
    out << "\n";
    out << superstruc[sc].conf.size() << " configurations in this supercell\n";
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      superstruc[sc].conf[nc].conc.print_concentration(out);
      superstruc[sc].conf[nc].print_bit(out);
    }
  }

}

//************************************************************

void configurations::read_energy_and_corr(){    // added by jishnu

  //first open a log file for mismatching names
  string log_file="energy_names.log";
  ofstream enname;
  enname.open(log_file.c_str());
  if(!enname){
    cout << "cannot open " << log_file << "\n";
    return;
  }



  char buff[300];
  ifstream enin;
  ifstream corrin;
  if(!scandirectory(".","energy")) {
    cout << "No energy file fto read from \n";
    exit(1);
  }
  else {
    enin.open("energy");
  }
  if(!scandirectory(".","corr.in")) {
    cout << "No corr.in file fto read from \n";
    exit(1);
  }
  else {
    corrin.open("corr.in");
  }
  // collect info from energy file
  enin.getline(buff,299);
  double fe,wt,co,dfh;
  vector <double> coo;
  string nm;vector <double> vecfe,vecwt,vecdfh;
  vector< vector<double> > veccoo;
  vector<string> vecnm;
  while(!enin.eof()) {
    enin >> fe >> wt;
    vecfe.push_back(fe);
    vecwt.push_back(wt);
    for (int i=0; i<superstruc[0].conf[0].coordinate.size()-1; i++) {
      enin >> co;
      coo.push_back(co);
    }
    veccoo.push_back(coo);
    coo.clear();
    enin >> dfh;
    vecdfh.push_back(dfh);
    enin >> nm;
    if(nm.size() > 1) vecnm.push_back(nm);
    nm.erase();
    enin.getline(buff,299);
  }
  //collect info from corr.in file
  int neci,nconf;
  corrin >> neci;
  corrin.getline(buff,299);
  corrin >> nconf;
  corrin.getline(buff,299);
  corrin.getline(buff,299);
  double elem;
  vector <double> cor;
  vector < vector<double> > veccor;
  for(int i=0;i<nconf;i++) {
    for (int j=0; j<neci; j++) {
      corrin >> elem;
      cor.push_back(elem);
    }
    veccor.push_back(cor);
    cor.clear();
  }
  // check if the energy and corr.in files are compatible to each other
  cout << "vecnm.size() =" << vecnm.size() << "\n";
  cout << "veccor.size() =" << veccor.size() << "\n";
  if(vecnm.size() != veccor.size()) {
    cout << " The size of the corr.in file is not same as the size in energy file ; Please check what's wrong. \n";
  }

  //put these info in superstruc	 -- new approach
  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      int index = -1;
      for(int nv=0;nv<vecnm.size();nv++) {
	if(superstruc[sc].conf[nc].name == vecnm[nv]) index = nv;
	if(index != -1) break;
      }
      if(index != -1){
	superstruc[sc].conf[nc].calculated = true;
	superstruc[sc].conf[nc].fenergy = vecfe[index];
	superstruc[sc].conf[nc].fpfenergy = vecfe[index];
	superstruc[sc].conf[nc].weight = vecwt[index];
	superstruc[sc].conf[nc].correlations.clear();
	for(int i=0; i<veccor[index].size(); i++){
	  superstruc[sc].conf[nc].correlations.push_back(veccor[index][i]);
	}

	int i=0;
	for(int j=0; j<superstruc[sc].conf[nc].conc.occup.size(); j++){
	  double sum=0.0;
	  for(int k=0; k<superstruc[sc].conf[nc].conc.occup[j].size()-1; k++){
	    superstruc[sc].conf[nc].conc.occup[j][k] = veccoo[index][i];
	    sum=sum+superstruc[sc].conf[nc].conc.occup[j][k];
	    i++;
	  }
	  superstruc[sc].conf[nc].conc.occup[j][superstruc[sc].conf[nc].conc.occup[j].size()-1]=1.0-sum;
	}
	superstruc[sc].conf[nc].assemble_coordinate_fenergy();

      }
    }
  }

  string file_name="custom_structures";
  ifstream in_dir;
  in_dir.open(file_name.c_str());

  if(!in_dir){
    cout << "cannot open file " << file_name << "\n";
    return;
  }
  while(!in_dir.eof()){
    string dir_name;
    in_dir >> dir_name;
    if(dir_name.size() > 0){
      int index = -1;
      for(int nv=0;nv<vecnm.size();nv++) {
	if(dir_name == vecnm[nv]) index = nv;
	if(index != -1) break;
      }
      if(index != -1){
	superstructure tsup;
	arrangement tarr;
	tarr.calculated = true;
	tarr.name = vecnm[index];
	tarr.fenergy = vecfe[index];
	tarr.fpfenergy = vecfe[index];
	tarr.weight = vecwt[index];
	tarr.delE_from_facet = vecdfh[index];
	tarr.correlations.clear();
	for(int i=0; i<veccor[index].size(); i++){
	  tarr.correlations.push_back(veccor[index][i]);
	}

	for(int j=0; j<superstruc[0].conf[0].conc.compon.size(); j++){  // set the concentration.compon structure same as the genrated structure
	  vector <specie> dummy;
	  for(int k=0; k<superstruc[0].conf[0].conc.compon[j].size(); k++){
	    dummy.push_back(superstruc[0].conf[0].conc.compon[j][k]);
	  }
	  tarr.conc.compon.push_back(dummy);
	  dummy.clear();
	}
	for(int j=0; j<superstruc[0].conf[0].conc.occup.size(); j++){  // set the concentration.occup structure of the custom_structure same as the generated ones
	  vector <double> dummy;
	  for(int k=0; k<superstruc[0].conf[0].conc.occup[j].size(); k++){
	    dummy.push_back(0.0);
	  }
	  tarr.conc.occup.push_back(dummy);
	  dummy.clear();
	}

	int i=0;
	for(int j=0; j<tarr.conc.occup.size(); j++){
	  double sum=0.0;
	  for(int k=0; k<tarr.conc.occup[j].size()-1; k++){
	    tarr.conc.occup[j][k] = veccoo[index][i];
	    sum=sum+tarr.conc.occup[j][k];
	    i++;
	  }
	  tarr.conc.occup[j][tarr.conc.occup[j].size()-1]=1.0-sum;
	}
	tarr.assemble_coordinate_fenergy();
	tsup.conf.push_back(tarr);
	superstruc.push_back(tsup);
      }
    }
  }


  in_dir.close();
  enname.close();

}
//************************************************************

void configurations::print_corr_old(){
  int i,j,k;
  int num_basis=0;
  int num_conf=0;

  ofstream out;
  out.open("CON.CORR");
  if(!out){
    cout << "cannot open CON.CORR \n";
    return;
  }

  for(i=0; i<basiplet.orb.size(); i++)
    num_basis=num_basis+basiplet.orb[i].size();

  for(i=0; i<superstruc.size(); i++)
    num_conf=num_conf+superstruc[i].conf.size();

  out << num_basis << "  number of basis function correlations\n";
  out << num_conf << "  number of configurations\n";
  out << "correlations \n";
  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++)
      superstruc[sc].conf[nc].print_correlations(out);
  }
}




//************************************************************

void configurations::print_con(){
  int i,j,k;

  ofstream out;
  out.open("configuration");
  if(!out){
    cout << "cannot open configuration \n";
    return;
  }

  out << "Structures generated within supercells \n";
  out << "\n";
  out << superstruc.size() << "  supercells considered\n";
  for(int sc=0; sc<superstruc.size(); sc++){
    out << "\n";
    for(i=0; i<3; i++){
      for(j=0; j<3; j++){
	out.precision(5);out.width(5);
	out << superstruc[sc].struc.slat[i][j] << " ";
      }
      out << "  ";
    }
    out << "\n";
    out << superstruc[sc].conf.size() << " configurations in this supercell\n";
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      out << superstruc[sc].conf[nc].name << "  ";  // added by jishnu
      superstruc[sc].conf[nc].conc.print_concentration_without_names(out);
      superstruc[sc].conf[nc].print_bit(out);
    }
  }

}

//************************************************************

void configurations::read_con(){  // added by jishnu (this s/r)

  string junk;
  double value;


  //----------------
  //test if prim sturcture is there or not
  if(prim.atom.size()==0){
    cout << "No prim structure read \n";
    return;
  }
  //----------------

  ifstream in;
  in.open("configuration");
  if(!in){
    cout << "cannot open configuration \n";
    return;
  }

  read_junk(in);
  int superstruc_size;
  in >> superstruc_size;
  read_junk(in);
  for(int sc=0;sc<superstruc_size;sc++){
    superstructure tsuperstructure;
    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	in >> tsuperstructure.struc.slat[i][j];
      }
    }
    tsuperstructure.struc.generate_lat(prim);
    tsuperstructure.struc.expand_prim_basis(prim);

    int conf_size;
    in >> conf_size;
    read_junk(in);
    for (int nc=0;nc<conf_size;nc++){
      arrangement tarrangement;
      tarrangement.conc.collect_components(prim);
      in >> tarrangement.name;
      tarrangement.conc.get_occup(in);
      tarrangement.get_bit(in);
      tsuperstructure.conf.push_back(tarrangement);
    }
    superstruc.push_back(tsuperstructure);
  }

}   // end of s/r read_con()

//************************************************************

void configurations::print_corr(){
  int i,j,k;
  int num_basis=0;
  int num_conf=0;

  ofstream out;
  out.open("configuration.corr");
  if(!out){
    cout << "cannot open configuration.corr \n";
    return;
  }

  for(i=0; i<basiplet.orb.size(); i++)
    num_basis=num_basis+basiplet.orb[i].size();

  for(i=0; i<superstruc.size(); i++)
    num_conf=num_conf+superstruc[i].conf.size();

  out << num_basis << "  number of basis function correlations\n";
  out << num_conf << "  number of configurations\n";
  out << "correlations \n";
  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      out << superstruc[sc].conf[nc].name;  // added by jishnu
      superstruc[sc].conf[nc].print_correlations(out);
    }
  }
}  // end of s/r
//************************************************************

void configurations::read_corr(){  // added by jishnu (this s/r) // this must be called after read_con

  string junk;
  int corr_size;

  ifstream in;
  in.open("configuration.corr");
  if(!in){
    cout << "cannot open configuration.corr \n";
    return;
  }

  in >> corr_size;
  for(int i=0;i<3;i++){
    read_junk(in);
  }
  double value;
  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      in >> junk;  // the name of the configuration is aleady read in from configurations file , so no need to do that again or overwrite
      for(int i=0;i<corr_size;i++){
	in >> value;
	superstruc[sc].conf[nc].correlations.push_back(value);
      }
    }
  }

}  // end of the s/r

//************************************************************
void configurations::reconstruct_from_read_files(){   // added by jishnu

  read_con();
  read_corr();

}   // end of the s/r
//************************************************************

void hull::clear_arrays(){   // added by jishnu
  point.clear();
  face.clear();
}
//************************************************************

void configurations::print_make_dirs(){   // modified by jishnu
  if(!scandirectory(".","make_dirs")){
    ofstream out("make_dirs");
    out << "#    name      make      concentrations  \n";

    for(int ns=0; ns<superstruc.size(); ns++){
      for(int nc=0; nc<superstruc[ns].conf.size(); nc++){
	out << superstruc[ns].conf[nc].name;
	if(superstruc[ns].conf[nc].make) out << "  1     ";
	else out << "  0     ";
	superstruc[ns].conf[nc].assemble_coordinate_fenergy();
	for(int i=0; i<superstruc[ns].conf[nc].coordinate.size()-1; i++){
	  out << superstruc[ns].conf[nc].coordinate[i] << "  ";
	}
	out << " \n";
      }
    }
    out.close();
  }


}


//************************************************************

void configurations::read_make_dirs(){   // modified by jishnu

  ifstream in("make_dirs");
  if(!in){
    cout << "cannot open file make_dirs\n";
    return;
  }
  char buff[300];
  in.getline(buff,299);

  while(!in.eof()){
    string struc_name;
    // double weight;
    int make;
    in >> struc_name;
    if(struc_name.size() > 0){
      // in >> weight;
      in >> make;
      in.getline(buff,299);

      //among all the structures in the superstructure vector, find that with the same name

      bool match=false;
      if(make){
	for(int ns=0; ns<superstruc.size(); ns++){
	  for(int nc=0; nc<superstruc[ns].conf.size(); nc++){
	    if(struc_name.compare(superstruc[ns].conf[nc].name) == 0){
	      match=true;
	      //superstruc[ns].conf[nc].weight=weight;
	      //if(make == 1) superstruc[ns].conf[nc].make=true;
	      //else superstruc[ns].conf[nc].make=false;
	      superstruc[ns].conf[nc].make=true;
	      break;
	    }
	  }
	  if(match)break;
	}
      }
    }


  }
  in.close();

  //remove the make_dirs file so that the most recent weights can be written

  int s=system("rm make_dirs");
  if(s == -1){
    cout << "was unable to remove make_dirs \n";
  }

}


//************************************************************

void configurations::collect_reference(){

  // reference states are those structures where either all concentration variables are zero or
  // at most one concentration variable is 1.0

  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){

      // go through the concentration object for each arrangement conf[] and search for
      // those in which all concentrations are either zero, or all are zero except for one

      int zero=0;
      int one=0;
      int total=0;
      for(int k=0; k<superstruc[sc].conf[nc].conc.occup.size(); k++){
	for(int l=0; l<superstruc[sc].conf[nc].conc.occup[k].size()-1; l++){
	  total++;
	  if(abs(superstruc[sc].conf[nc].conc.occup[k][l]) < tol)zero++;
	  if(abs(superstruc[sc].conf[nc].conc.occup[k][l]-1.0) < tol)one++;
	}
      }
      if(zero == total){
	reference.push_back(superstruc[sc].conf[nc]);
      }
      if(zero == total-1 && one == 1){
	// this part added by jishnu
	int count=0;
	for(int er=0; er<reference.size(); er++){
	  if(!compare(superstruc[sc].conf[nc].correlations,reference[er].correlations)) count++;
	  else{
	    if(superstruc[sc].conf[nc].calculated && reference[er].calculated) {
	      double difference=superstruc[sc].conf[nc].energy-reference[er].energy;
	      if(difference >=tol) {cout << "CAUTION!! The reference structures are same but the energies are significantly different and we are not adding the new reference\n"; }
	    }
	    else if(superstruc[sc].conf[nc].calculated && !reference[er].calculated) reference[er]=superstruc[sc].conf[nc];
	  }
	}
	if(count == reference.size())   reference.push_back(superstruc[sc].conf[nc]);
	// end of added by jishnu
      }
    }
  }

  if(!scandirectory(".","reference")){
    ofstream out("reference");

    out << "concentration and energy of the reference states\n";
    out << "\n";

    for(int i=0; i<reference.size(); i++){
      out << "reference " << i << "  ";
      out <<  reference[i].name << "\n"; // added by jishnu
      for(int ii=0; ii<reference[i].conc.compon.size(); ii++){
	out << "sublattice " << ii << "\n";
	for(int j=0; j<reference[i].conc.compon[ii].size(); j++){
	  for(int k=0; k<2; k++)
	    out << reference[i].conc.compon[ii][j].name[k];
	  out << "  ";
	}
	out << "\n";

	for(int j=0; j<reference[i].conc.compon[ii].size(); j++)
	  out << reference[i].conc.occup[ii][j] << "  ";
	out << "\n";

	out << reference[i].energy << "  energy \n";
      }
      out << "\n";

    }
    out.close();
  }

  ifstream in("reference");

  char buff[200];
  in.getline(buff,199);

  for(int i=0; i<reference.size(); i++){
    if(in.eof()){
      cout << "reference file is not compatible with current system\n";
      cout << "using conventional reference \n";
      return;
    }
    in.getline(buff,199);
    in.getline(buff,199);
    for(int ii=0; ii<reference[i].conc.compon.size(); ii++){
      if(in.eof()){
	cout << "reference file is not compatible with current system\n";
	cout << "using conventional reference \n";
	return;
      }
      in.getline(buff,199);
      in.getline(buff,199);
      for(int j=0; j<reference[i].conc.compon[ii].size(); j++){
	in >> reference[i].conc.occup[ii][j];
      }
      in.getline(buff,199);
      in >> reference[i].energy;
      in.getline(buff,199);
    }
  }
}


//************************************************************

void configurations::collect_energies(){

  //first open a log file with problematic relaxations
  string log_file="relax.log";
  ofstream relaxlog;
  relaxlog.open(log_file.c_str());
  if(!relaxlog){
    cout << "cannot open " << log_file << "\n";
    return;
  }


  for(int sc=0; sc<superstruc.size(); sc++){
    multiplet super_basiplet;
    superstruc[sc].struc.expand_prim_clust(basiplet,super_basiplet);

    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(scandirectory(".",superstruc[sc].conf[nc].name)){
	if(scandirectory(superstruc[sc].conf[nc].name, "OSZICAR")){

	  double energy;
	  int relax_step;    // added by jishnu

	  //extract the energy from the OSZICAR file in directory= superstruc[sc].conf[nc].name

	  // if(read_oszicar(superstruc[sc].conf[nc].name, energy)){
	  if(read_oszicar(superstruc[sc].conf[nc].name, energy, relax_step)){   // changed by jishnu // used overloaded function to count the no of relax step

	    //normalize the energy by the number of primitive unit cells

	    double vol=determinant(superstruc[sc].struc.slat);
	    if(abs(vol) > tol){
	      superstruc[sc].conf[nc].energy= energy/abs(vol);
	      superstruc[sc].conf[nc].calculated = true;
	      superstruc[sc].conf[nc].relax_step = relax_step;    // added by jishnu

	      //check first if there is a POS file, if not, create it for this configuration
	      structure relaxed;
	      relaxed.collect_relax(superstruc[sc].conf[nc].name);

	      double rescale=pow(1.0/determinant(relaxed.strain),1.0/3.0);

	      relaxed.generate_slat(prim,rescale);

	      if(!compare(relaxed.slat,superstruc[sc].struc.slat)){
		relaxlog << superstruc[sc].conf[nc].name << " the relaxed cell is not the same supercell as the original supercell\n";
	      }
	      else{
		arrangement relaxed_conf;
		relaxed.map_on_expanded_prim_basis(prim,relaxed_conf);
		calc_correlations(relaxed, super_basiplet, relaxed_conf);
		if(!compare(relaxed_conf.correlations,superstruc[sc].conf[nc].correlations)){
		  relaxlog << superstruc[sc].conf[nc].name << " relaxed to a new structure\n";

		  //----- added by jishnu ----------
		  for(int ncc=0; ncc<superstruc[sc].conf.size(); ncc++){
		    if(compare(relaxed_conf.correlations,superstruc[sc].conf[ncc].correlations)){
		      relaxlog << relaxed_conf.name << " and is a duplicate of " << superstruc[sc].conf[ncc].name << "\n";
		      break;
		    }
		  }
		  //----- added by jishnu ----------

		  superstruc[sc].conf[nc].weight=0.0;
		  for(int i=0; i<superstruc[sc].conf[nc].correlations.size(); i++){
		    superstruc[sc].conf[nc].correlations[i]=relaxed_conf.correlations[i];
		  }
		}
		else{
		  superstruc[sc].conf[nc].weight=1.0;
		}
	      }
	    }
	  }
	}
      }

    }
  }

  //Read in the names of directories of manually made configurations, collect them and add them to the
  //configs object

  // -read in the POSCAR and the CONTCAR, determine slat, and compare to all supercells already found
  // - also compare the slat from the POSCAR with that from the CONTCAR
  // -determine the arrangement and add it to that structure (if it is not already included)
  // -if the arrangement already exists, keep a log of overlapping structures



  string file_name="custom_structures";
  ifstream in_dir;
  in_dir.open(file_name.c_str());

  if(!in_dir){
    cout << "cannot open file " << file_name << "\n";
    return;
  }


  while(!in_dir.eof()){
    string dir_name;
    in_dir >> dir_name;
    if(dir_name.size() > 0){

      cout << "WORKING ON " << dir_name << "\n";

      //first check whether the original POS can be mapped onto a supercell of PRIM


      if(scandirectory(dir_name,"POS")){

	structure prerelaxed;

	string pos_file=dir_name;
	pos_file.append("/POS");
	ifstream in_pos;
	in_pos.open(pos_file.c_str());
	if(!in_pos){
	  cout << "cannot open file " << in_pos << "\n";
	  return;
	}


	prerelaxed.read_struc_poscar(in_pos);

	prerelaxed.generate_slat(prim);

	prerelaxed.idealize();
	arrangement prerelaxed_conf;
	prerelaxed_conf.name=dir_name;
	prerelaxed.map_on_expanded_prim_basis(prim, prerelaxed_conf);

	prerelaxed_conf.conc.collect_components(prim);
	prerelaxed_conf.conc.calc_concentration(prerelaxed);

	//	prerelaxed.map_on_expanded_prim_basis(prim, prerelaxed_conf);


	multiplet super_basiplet;

	prerelaxed.expand_prim_clust(basiplet,super_basiplet);

	calc_correlations(prerelaxed, super_basiplet, prerelaxed_conf);

	//read the energy from the OSZICAR file if it exists

	if(scandirectory(dir_name,"OSZICAR")){
	  double energy;
	  int relax_step; // added by jishnu
	  // if(read_oszicar(dir_name,energy)){
	  if(read_oszicar(dir_name, energy, relax_step)){   // changed by jishnu // used overloaded function to count the no of relax step
	    double vol=determinant(prerelaxed.slat);
	    if(abs(vol) > tol){
	      prerelaxed_conf.energy=energy/abs(vol);
	      prerelaxed_conf.calculated = true;
	      prerelaxed_conf.relax_step = relax_step;   // added by jishnu
	    }
	  }
	}

	//if there is a CONTCAR - read it and compare slat and the configuration

	if(scandirectory(dir_name,"CONTCAR")){
	  structure relaxed;
	  relaxed.collect_relax(dir_name);
	  double rescale=pow(1.0/determinant(relaxed.strain),1.0/3.0);
	  relaxed.generate_slat(prim,rescale);


	  if(!compare(relaxed.slat,prerelaxed.slat)){
	    relaxlog << prerelaxed_conf.name << " the relaxed cell is not the same supercell as the original supercell\n";
	  }
	  else{
	    arrangement relaxed_conf;
	    relaxed.map_on_expanded_prim_basis(prim,relaxed_conf);
	    calc_correlations(relaxed, super_basiplet, relaxed_conf);
	    if(!compare(relaxed_conf.correlations,prerelaxed_conf.correlations)){
	      relaxlog << prerelaxed_conf.name << " relaxed to a new structure\n";
	      for(int i=0; i< prerelaxed_conf.correlations.size(); i++){
		prerelaxed_conf.correlations[i]=relaxed_conf.correlations[i];
	      }
	    }
	  }
	}

	//go through all current superstructures from configs and see if this supercell is already there
	//if not, add the super cell
	//else see if the conf is already there
	//if not add it, other wise make a note in the log file
	//collect the energy

	int non_match_sc=0;
	for(int sc=0; sc<superstruc.size(); sc++){
	  if(compare(prerelaxed.slat,superstruc[sc].struc.slat)){

	    //see if the configuration already exists
	    int non_match_nc=0;
	    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
	      if(compare(prerelaxed_conf.correlations,superstruc[sc].conf[nc].correlations)){
		relaxlog << prerelaxed_conf.name << " is a duplicate of " << superstruc[sc].conf[nc].name << "\n";
		//Edited by jishnu
		if((superstruc[sc].conf[nc].calculated)&& (prerelaxed_conf.calculated)){
		  if(abs(superstruc[sc].conf[nc].energy-prerelaxed_conf.energy) > tol) {
		    relaxlog << "CAUTION!!!!  The energy of custom structure " << prerelaxed_conf.name <<" is diffferent from its duplicate structure (" << superstruc[sc].conf[nc].name
			     << ") by " << prerelaxed_conf.energy-superstruc[sc].conf[nc].energy << " eV.\n";
		    cout << "CAUTION!!!!  The energy of custom structure " << prerelaxed_conf.name <<" is diffferent from its duplicate structure (" << superstruc[sc].conf[nc].name
			 << ") by " << prerelaxed_conf.energy-superstruc[sc].conf[nc].energy << " eV.\n";

		  }
		}
		if((!superstruc[sc].conf[nc].calculated) && (prerelaxed_conf.calculated)){
		  superstruc[sc].conf[nc].name=prerelaxed_conf.name;  // The custom structure replaces generated structure
		  superstruc[sc].conf[nc].energy=prerelaxed_conf.energy;
		  superstruc[sc].conf[nc].calculated=true;
		}
		break;
	      }
	      else{
		non_match_nc++;
	      }
	    }
	    if(non_match_nc == superstruc[sc].conf.size()){
	      cout << "THIS STRUCTURE IS UNIQUE - we are including it \n";
	      cout << "It's name is " << prerelaxed_conf.name << " \n";
	      cout << "\n";
	      prerelaxed_conf.ns=sc;
	      prerelaxed_conf.nc=superstruc[sc].conf.size();
	      superstruc[sc].conf.push_back(prerelaxed_conf);
	      break;
	    }
	  }
	  else{
	    non_match_sc++;
	  }
	}
	if(non_match_sc == superstruc.size()){
	  cout << "THIS STRUCTURE IS UNIQUE - we are including it \n";
	  cout << "It's name is " << prerelaxed_conf.name << " \n";
	  cout << "\n";
	  superstructure tsuperstruc;
	  tsuperstruc.struc=prerelaxed;
	  prerelaxed_conf.ns=superstruc.size();
	  prerelaxed_conf.nc=0;
	  superstruc.push_back(tsuperstruc);
	  superstruc[non_match_sc].conf.push_back(prerelaxed_conf);
	}

      }

    }
  }


  in_dir.close();
  relaxlog.close();


  //read in the weights for all these structures



}

//************************************************************

void configurations::collect_energies_fast(){

  //first open a log file with problematic relaxations
  string log_file="relax.log";
  ofstream relaxlog;
  relaxlog.open(log_file.c_str());
  if(!relaxlog){
    cout << "cannot open " << log_file << "\n";
    return;
  }

  double tclust_func, tcorr;

  for(int sc=0; sc<superstruc.size(); sc++){

    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(scandirectory(".",superstruc[sc].conf[nc].name)){
	if(scandirectory(superstruc[sc].conf[nc].name, "OSZICAR")){
	  if(!(superstruc[sc].corr_to_atom_vec.size() && superstruc[sc].corr_to_atom_vec.size())){
	    //If cluster function and basis information do not exist for this supercell, populate the vectors
	    multiplet super_basiplet;
	    superstruc[sc].struc.expand_prim_clust(basiplet,super_basiplet);
	    get_corr_vector(superstruc[sc].struc, super_basiplet, superstruc[sc].corr_to_atom_vec);
	    get_super_basis_vec(superstruc[sc].struc, superstruc[sc].basis_to_bit_vec);
	  }

	  double energy;
	  int relax_step;    // added by jishnu

	  //extract the energy from the OSZICAR file in directory= superstruc[sc].conf[nc].name
	  // if(read_oszicar(superstruc[sc].conf[nc].name, energy)){
	  if(read_oszicar(superstruc[sc].conf[nc].name, energy, relax_step)){   // changed by jishnu // used overloaded function to count the no of relax step

	    //normalize the energy by the number of primitive unit cells

	    double vol=determinant(superstruc[sc].struc.slat);
	    if(abs(vol) > tol){
	      superstruc[sc].conf[nc].energy= energy/abs(vol);
	      superstruc[sc].conf[nc].calculated = true;
	      superstruc[sc].conf[nc].relax_step = relax_step;    // added by jishnu

	      //check first if there is a POS file, if not, create it for this configuration
	      structure relaxed;
	      relaxed.collect_relax(superstruc[sc].conf[nc].name);

	      double rescale=pow(1.0/determinant(relaxed.strain),1.0/3.0);

	      relaxed.generate_slat(prim,rescale);

	      if(!compare(relaxed.slat,superstruc[sc].struc.slat)){
		relaxlog << superstruc[sc].conf[nc].name << " the relaxed cell is not the same supercell as the original supercell\n";
	      }
	      else{
		arrangement relaxed_conf;
		relaxed.map_on_expanded_prim_basis(prim,relaxed_conf);

		//Calculate correlations
		relaxed_conf.correlations.push_back(1.0);
		int atom_ind, bit_ind;
		for(int i=0; i<superstruc[sc].corr_to_atom_vec.size(); i++){
		  tcorr=0.0;
		  for(int j=0; j<superstruc[sc].corr_to_atom_vec[i].size(); j++){
		    tclust_func=1.0;
		    for(int k=0; k<superstruc[sc].corr_to_atom_vec[i][j].size(); k++){
		      atom_ind=superstruc[sc].corr_to_atom_vec[i][j][k][0];
		      bit_ind=superstruc[sc].corr_to_atom_vec[i][j][k][1];
		      tclust_func*=superstruc[sc].basis_to_bit_vec[atom_ind][relaxed_conf.bit[atom_ind]][bit_ind];
		    }
		    tcorr+=tclust_func;
		  }
		  relaxed_conf.correlations.push_back(tcorr/superstruc[sc].corr_to_atom_vec[i].size());
		}

		if(!compare(relaxed_conf.correlations,superstruc[sc].conf[nc].correlations)){
		  relaxlog << superstruc[sc].conf[nc].name << " relaxed to a new structure\n";

		  //----- added by jishnu ----------
		  for(int ncc=0; ncc<superstruc[sc].conf.size(); ncc++){
		    if(compare(relaxed_conf.correlations,superstruc[sc].conf[ncc].correlations)){
		      relaxlog << relaxed_conf.name << " and is a duplicate of " << superstruc[sc].conf[ncc].name << "\n";
		      break;
		    }
		  }
		  //----- added by jishnu ----------

		  superstruc[sc].conf[nc].weight=0.0;
		  for(int i=0; i<superstruc[sc].conf[nc].correlations.size(); i++){
		    superstruc[sc].conf[nc].correlations[i]=relaxed_conf.correlations[i];
		  }
		}
		else{
		  superstruc[sc].conf[nc].weight=1.0;
		}
	      }
	    }
	  }
	}
      }

    }
  }

  //Read in the names of directories of manually made configurations, collect them and add them to the
  //configs object

  // -read in the POSCAR and the CONTCAR, determine slat, and compare to all supercells already found
  // - also compare the slat from the POSCAR with that from the CONTCAR
  // -determine the arrangement and add it to that structure (if it is not already included)
  // -if the arrangement already exists, keep a log of overlapping structures



  string file_name="custom_structures";
  ifstream in_dir;
  in_dir.open(file_name.c_str());

  if(!in_dir){
    cout << "cannot open file " << file_name << "\n";
    return;
  }


  while(!in_dir.eof()){
    string dir_name;
    in_dir >> dir_name;
    if(dir_name.size() > 0){

      cout << "WORKING ON " << dir_name << "\n";

      //first check whether the original POS can be mapped onto a supercell of PRIM


      if(scandirectory(dir_name,"POS")){

	structure prerelaxed;

	string pos_file=dir_name;
	pos_file.append("/POS");
	ifstream in_pos;
	in_pos.open(pos_file.c_str());
	if(!in_pos){
	  cout << "cannot open file " << in_pos << "\n";
	  return;
	}


	prerelaxed.read_struc_poscar(in_pos);
	prerelaxed.generate_slat(prim);
	prerelaxed.idealize();
	arrangement prerelaxed_conf;
	prerelaxed_conf.name=dir_name;
	prerelaxed.map_on_expanded_prim_basis(prim, prerelaxed_conf);

	prerelaxed_conf.conc.collect_components(prim);
	prerelaxed_conf.conc.calc_concentration(prerelaxed);


	//Edited by John - find supercell number first, ensure existence of corr_to_atom_vec and basis_to_bit_vec
	int sc_ind=-1;
	for(int sc=0; sc<superstruc.size(); sc++){
	  if(compare(prerelaxed.slat,superstruc[sc].struc.slat)){
	    sc_ind=sc;
	    break;
	  }
	}
	if(sc_ind==-1){
	  // cout << "New supercell encountered...";
	  sc_ind=superstruc.size();

	  superstructure tsuperstruc;
	  tsuperstruc.struc=prerelaxed;
	  superstruc.push_back(tsuperstruc);
	  // cout << "Added. \n";
	}

	int atom_ind, bit_ind;

	if(!(superstruc[sc_ind].corr_to_atom_vec.size() && superstruc[sc_ind].basis_to_bit_vec.size())){
	  // cout << "Cluster calculation vectors not present for this supercell... ";
	  multiplet super_basiplet;
	  superstruc[sc_ind].struc.expand_prim_clust(basiplet,super_basiplet);
	  get_corr_vector(superstruc[sc_ind].struc, super_basiplet, superstruc[sc_ind].corr_to_atom_vec);
	  get_super_basis_vec(superstruc[sc_ind].struc, superstruc[sc_ind].basis_to_bit_vec);
	  // cout << "Added.\n"
	}

	//Calculate correlations of POS
	prerelaxed_conf.correlations.push_back(1.0);
	for(int i=0; i<superstruc[sc_ind].corr_to_atom_vec.size(); i++){
	  tcorr=0.0;
	  for(int j=0; j<superstruc[sc_ind].corr_to_atom_vec[i].size(); j++){
	    tclust_func=1.0;
	    for(int k=0; k<superstruc[sc_ind].corr_to_atom_vec[i][j].size(); k++){
	      atom_ind=superstruc[sc_ind].corr_to_atom_vec[i][j][k][0];
	      bit_ind=superstruc[sc_ind].corr_to_atom_vec[i][j][k][1];
	      tclust_func*=superstruc[sc_ind].basis_to_bit_vec[atom_ind][prerelaxed_conf.bit[atom_ind]][bit_ind];
	    }
	    tcorr+=tclust_func;
	  }
	  prerelaxed_conf.correlations.push_back(tcorr/superstruc[sc_ind].corr_to_atom_vec[i].size());
	}

	//\End edit by John

	//read the energy from the OSZICAR file if it exists

	if(scandirectory(dir_name,"OSZICAR")){
	  double energy;
	  int relax_step; // added by jishnu
	  // if(read_oszicar(dir_name,energy)){
	  if(read_oszicar(dir_name, energy, relax_step)){   // changed by jishnu // used overloaded function to count the no of relax step
	    double vol=determinant(prerelaxed.slat);
	    if(abs(vol) > tol){
	      prerelaxed_conf.energy=energy/abs(vol);
	      prerelaxed_conf.calculated = true;
	      prerelaxed_conf.relax_step = relax_step;   // added by jishnu
	    }
	  }
	}

	//if there is a CONTCAR - read it and compare slat and the configuration

	if(scandirectory(dir_name,"CONTCAR")){
	  structure relaxed;
	  relaxed.collect_relax(dir_name);
	  double rescale=pow(1.0/determinant(relaxed.strain),1.0/3.0);
	  relaxed.generate_slat(prim,rescale);


	  if(!compare(relaxed.slat,prerelaxed.slat)){
	    relaxlog << prerelaxed_conf.name << " the relaxed cell is not the same supercell as the original supercell\n";
	  }
	  else{
	    arrangement relaxed_conf;
	    relaxed.map_on_expanded_prim_basis(prim,relaxed_conf);

	    //Calculate correlations of CONTCAR
	    relaxed_conf.correlations.push_back(1.0);
	    for(int i=0; i<superstruc[sc_ind].corr_to_atom_vec.size(); i++){
	      tcorr=0.0;
	      for(int j=0; j<superstruc[sc_ind].corr_to_atom_vec[i].size(); j++){
		tclust_func=1.0;
		for(int k=0; k<superstruc[sc_ind].corr_to_atom_vec[i][j].size(); k++){
		  atom_ind=superstruc[sc_ind].corr_to_atom_vec[i][j][k][0];
		  bit_ind=superstruc[sc_ind].corr_to_atom_vec[i][j][k][1];
		  tclust_func*=superstruc[sc_ind].basis_to_bit_vec[atom_ind][relaxed_conf.bit[atom_ind]][bit_ind];
		}
		tcorr+=tclust_func;
	      }
	      relaxed_conf.correlations.push_back(tcorr/superstruc[sc_ind].corr_to_atom_vec[i].size());
	    }


	    if(!compare(relaxed_conf.correlations,prerelaxed_conf.correlations)){
	      relaxlog << prerelaxed_conf.name << " relaxed to a new structure\n";
	      for(int i=0; i< prerelaxed_conf.correlations.size(); i++){
		prerelaxed_conf.correlations[i]=relaxed_conf.correlations[i];
	      }
	    }

	  }
	}

	//go through all current superstructures and configs and see if this
	//configuration is already there.  If not, add the configuration.
	//other wise make a note in the log file
	//collect the energy

	//see if the configuration already exists
	bool new_flag=true;
	for(int ns=0; ns<superstruc.size(); ns++){
	  for(int nc=0; nc<superstruc[ns].conf.size(); nc++){
	    if(compare(prerelaxed_conf.correlations,superstruc[ns].conf[nc].correlations)){
	      new_flag=false;
	      relaxlog << prerelaxed_conf.name << " is a duplicate of " << superstruc[ns].conf[nc].name << "\n";
	      //Edited by jishnu
	      if((superstruc[ns].conf[nc].calculated)&& (prerelaxed_conf.calculated)){
		if(abs(superstruc[ns].conf[nc].energy-prerelaxed_conf.energy) > tol) {
		  relaxlog << "CAUTION!!!!  The energy of custom structure (" << prerelaxed_conf.name
			   <<") is very diffferent from that of the generated structure("
			   << superstruc[ns].conf[nc].name << ")\n";
		  relaxlog << "   Energy of " << prerelaxed_conf.name << " differs by "
			   << prerelaxed_conf.energy-superstruc[ns].conf[nc].energy
			   << " eV.\n";
		  cout << "CAUTION!!!!  The energy custom structure (" << prerelaxed_conf.name
		       <<") is very diffferent from that of the generated structure("
		       << superstruc[ns].conf[nc].name << ")\n";
		  cout << "   Energy of " << prerelaxed_conf.name << " differs by "
		       << prerelaxed_conf.energy-superstruc[ns].conf[nc].energy
		       << " eV.\n";
		}
	      }
	      if((!superstruc[ns].conf[nc].calculated) && (prerelaxed_conf.calculated)){
		superstruc[ns].conf[nc].name=prerelaxed_conf.name;  // The custom structure replaces generated structure
		superstruc[ns].conf[nc].energy=prerelaxed_conf.energy;
		superstruc[ns].conf[nc].calculated=true;
	      }
	      // break;  //Commented by John.  could cause issues if more than two structures are identical.
	    }
	  }
	}
	if(new_flag){
	  cout << "THIS STRUCTURE IS UNIQUE - we are including it \n";
	  cout << "Its name is " << prerelaxed_conf.name << " \n";
	  cout << "\n";
	  prerelaxed_conf.ns=sc_ind;
	  prerelaxed_conf.nc=superstruc[sc_ind].conf.size();
	  superstruc[sc_ind].conf.push_back(prerelaxed_conf);
	}
      }

    }
  }


  in_dir.close();
  relaxlog.close();


  //read in the weights for all these structures



}


//************************************************************
//************************************************************

void configurations::calculate_formation_energy(){  // modified by jishnu

  facet ref_plane;
  for(int i=0;i<reference.size();i++)	ref_plane.corner.push_back(reference[i]);
  for(int i=0;i<reference.size();i++) ref_plane.corner[i].update_te();   // very IMPORTANT statement
  ref_plane.get_norm_vec();

  for(int sc=0; sc<superstruc.size();  sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(superstruc[sc].conf[nc].calculated){
	superstruc[sc].conf[nc].update_te();	// again very IMPORTANT
	superstruc[sc].conf[nc].got_fenergy = ref_plane.find_endiff(superstruc[sc].conf[nc],superstruc[sc].conf[nc].fpfenergy); // here got_fenergy does not mean anything, whether it is true or false, the formation energy value is always accepted.
	superstruc[sc].conf[nc].got_fenergy = true; // here we are forcing it to be true
	superstruc[sc].conf[nc].update_fp();
      }
    }
  }
}

//************************************************************
void configurations::assemble_hull(){    // Modified by jishnu

  // of all calculated configurations, keep only the lowest energy one for each concentration
  // copy these into an array
  // feed the array to the hull finder
  // keep track of indexing

  vector<arrangement> tconf;
  for(int ns=0; ns<superstruc.size(); ns++){
    for(int nc=0; nc<superstruc[ns].conf.size(); nc++){
      if(superstruc[ns].conf[nc].calculated || ((superstruc[ns].conf[nc].ce == 1) && (superstruc[ns].conf[nc].fp == 0) && (superstruc[ns].conf[nc].te == 0))){                // first condition works for the fp hull and second condition works for the CE hull // jishnu

	//compare this config with the tconf already collected
	//if the concentration is already present, keep the one with the lowest energy
	//otherwise add this point to the list

	int i;
	for(i=0; i<tconf.size(); i++){
	  if(compare(superstruc[ns].conf[nc].conc,tconf[i].conc)){
	    if(superstruc[ns].conf[nc].fenergy < tconf[i].fenergy){
	      tconf[i]=superstruc[ns].conf[nc];
	      tconf[i].assemble_coordinate_fenergy();
	    }
	    break;
	  }
	}
	if(i == tconf.size()){
	  tconf.push_back(superstruc[ns].conf[nc]);
	  tconf[i].assemble_coordinate_fenergy();
	}
      }
    }
  }

  if(tconf.size() == 0){
    cout << "No configurations available to determine convex hull\n";
    cout << "quitting assemble_hull() \n";
    return;
  }


  //determine the number of independent concentration variables
  int dim=0;
  for(int i=0; i<tconf[0].conc.occup.size(); i++){
    for(int j=0; j<tconf[0].conc.occup[i].size()-1; j++){
      dim++;
    }
  }


  if(dim > 2){
    cout << "At this point we can only determine the convex hull for at most a ternary system\n";
    cout << "quitting assemble_hull() \n";
    return;
  }

  if (dim == 1) {   // i.e. if the system is binary

    double *matrix = new double [tconf.size()*3];  // matrix stores label, conc and FP energy
    for (int i=0;i<tconf.size();i++){
      matrix[i*3+0] = i;  // keeping track of the unique structure details
      matrix[i*3+1] = tconf[i].coordinate[0];
      matrix[i*3+2] = tconf[i].coordinate[1];

    }

    Array mat(tconf.size(),3);
    mat.setArray(matrix);
    Array gs(3,3);  // first index is no of rows,which can be anything, but second colum is no of columns which must be 3 for binary.
    Array edge(3,2); //again u can put any no instead of 3, but keep 2 fixed.
    gs = mat.hullnd(2,edge); // write the hull points to gs Array, the 1st column is label, 2nd concentration, 3rd E

    gs.assort_a(2);  // this sorts the gs array according to the i th column (starting from 1) // here it is concentration column

    int nr = gs.num_row();
    double *gs_all = new double [nr*3];
    for(int i=0;i<nr;i++){
      gs_all[i*3+0] = gs.elem(i+1,1);
      gs_all[i*3+1] = gs.elem(i+1,2);
      gs_all[i*3+2] = gs.elem(i+1,3);
    }  //gs_all is equivalent to matrix (before finding the hull), it contains all the hull points 	(label, concentration and energy in order)

    // finding the left and right end points of the binary hull
    double *left_end = new double [3];
    double *right_end = new double [3];
    for(int i=0;i<3;i++)	left_end[i]=gs_all[0+i];
    for(int i=0;i<3;i++)	right_end[i]=gs_all[0+i];

    for(int i=1;i<nr;i++){
      if (gs_all[i*3+1] < left_end[1]) for(int j=0;j<3;j++)	{left_end[j]=gs_all[i*3+j];}

      if (gs_all[i*3+1] > right_end[1]) for(int j=0;j<3;j++)	{right_end[j]=gs_all[i*3+j];}

    } // end of finding the ends

    // finding the striaght line joining the end points
    double slope = (right_end[2]-left_end[2])/(right_end[1]-left_end[1]);
    // double intercept = right_end[1] - slope*right_end[0]; // no need to calculate this
    // end of finding the straight line

    // finding the lower half of the hull
    for(int j=0;j<tconf.size();j++){  // first put the left end into the pool of hull points
      if(j == left_end[0]) chull.point.push_back(tconf[j]);
    }

    for (int i=0;i<nr;i++){
      if((gs_all[3*i+1]-left_end[1]) != 0.0){
	double slope_p=(gs_all[3*i+2]-left_end[2])/(gs_all[3*i+1]-left_end[1]);
	if(slope_p <= slope){
	  for(int j=0;j<tconf.size();j++){
	    if(j == gs_all[3*i+0]) chull.point.push_back(tconf[j]);
	  }
	}
      }
    }
    // end of finding and saving the lower half of the hull
    // save the facet info
    for (int i=0;i<(chull.point.size()-1);i++){   // "-1" is there because there will be 7 facets(or edges in case of binary) for 8 points
      facet tfacet;
      tfacet.corner.push_back(chull.point[i]);  // the points are already sorted according to the concentration
      tfacet.corner.push_back(chull.point[i+1]);
      chull.face.push_back(tfacet);
    }

  }  // end of dim == 1 loop

  if (dim == 2) {   // i.e. if the system is ternary

    double *matrix = new double [tconf.size()*4];  // matrix stores conc1, conc2, FP energy, and label
    for (int i=0;i<tconf.size();i++){
      matrix[i*4+0] = i;  // keeping track of the unique structure details
      matrix[i*4+1] = tconf[i].coordinate[0];
      matrix[i*4+2] = tconf[i].coordinate[1];
      matrix[i*4+3] = tconf[i].coordinate[2];

    }

    Array mat(tconf.size(),4);
    mat.setArray(matrix);
    mat.assort_a2(2,3);    //assort data by concentration in ascending order
    Array gs(3,4);  // first index is no of rows,which can be anything, but second colum is no of columns which must be 4 for ternary.
    Array fa(3,3); //again u can put any no instead of first 3, but keep second 3 fixed.
    gs = mat.half_hull3(fa); // write the hull points to gs Array, the 1st column is label, 2nd and 3rd concentration, 4th E // this has already eliminated the upper portion of the hull
    int nr = gs.num_row();
    int nf = fa.num_row();  // number of facets
    //cout << "no of facets = " << nf << "\n";

    for (int i=0;i<nr;i++){
      for (int j=0;j<tconf.size();j++){
	if(gs.elem(i+1,1) == j) chull.point.push_back(tconf[j]);
      }
    }

    // save the facet info
    for (int i=0;i<nf;i++){
      facet tfacet;
      for(int j=0;j<(dim+1);j++){
	for(int k=0;k<nr;k++){
	  if(fa.elem(i+1,j+1) == gs.elem(k+1,1)) tfacet.corner.push_back(tconf[gs.elem(k+1,1)]);
	}
      }
      chull.face.push_back(tfacet);
    }

  }  // end of dim == 2 loop



}     // end of assemble_hull
//************************************************************
//************************************************************
void configurations::get_delE_from_hull(){     // added by jishnu

  for (int i=0;i<chull.face.size();i++){
    chull.face[i].get_norm_vec();
  }

  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(superstruc[sc].conf[nc].calculated){
	for (int i=0;i<chull.face.size();i++){
	  if(chull.face[i].find_endiff(superstruc[sc].conf[nc],superstruc[sc].conf[nc].delE_from_facet)) {  // dont need to do anything becuase the correct one is the last one and that is the only delE_from_facet that will be saved.
	    break;   // there is no need to continue on i loop once u find the right facet // not only "no need" but also, the correct value of delE_from_facet will be overwritten
	  }
	}
      }
    }
  }


} // end of get_delE_from_hull
//************************************************************
//************************************************************
void configurations::get_delE_from_hull_w_clexen(){     // added by jishnu

  for (int i=0;i<chull.face.size();i++){
    chull.face[i].get_norm_vec();
  }

  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(superstruc[sc].conf[nc].got_cefenergy){
	int i;
	for (i=0;i<chull.face.size();i++){
	  if(chull.face[i].find_endiff(superstruc[sc].conf[nc],superstruc[sc].conf[nc].delE_from_facet)) {  // dont need to do anything becuase the correct one is the last one and that is the only delE_from_facet that will be saved.
	    break;   // there is no need to continue on i loop once u find the right facet // not only "no need" but also, the correct value of delE_from_facet will be overwritten
	  }
	}
      }
    }
  }


} // end of get_delE_from_clex_hull
//************************************************************

//************************************************************
void hull::write_clex_hull(){   // added by jishnu

  string hull_clex_file="hull.clex";
  ofstream hullclex;
  hullclex.open(hull_clex_file.c_str());
  if(!hullclex){
    cout << "cannot open hull.clex file.\n";
    return;
  }
  hullclex << "# formation_energy        meaningless          concentrations            meaningless            name \n";
  for(int i=0; i<point.size(); i++){
    hullclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << point[i].coordinate[point[i].coordinate.size()-1];
    hullclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << 0;
    for(int j=0;j<(point[i].coordinate.size()-1);j++){
      hullclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) <<  point[i].coordinate[j];
    }
    hullclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << 0;
    hullclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(15)<<setprecision(0) << point[i].name << "\n";
  }

  hullclex.close();


  string facet_clex_file="facet.clex";
  ofstream facettclex;
  facettclex.open(facet_clex_file.c_str());
  if(!facettclex){
    cout << "cannot open facet.clex file.\n";
    return;
  }

  for(int i=0;i<face.size();i++){
    for(int j=0;j<face[i].corner.size();j++){
      facettclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(15)<<setprecision(0)<< face[i].corner[j].name;
    }
    facettclex << "\n";
  }

  facettclex.close();


}
//************************************************************
//************************************************************

void configurations::cluster_expanded_energy(){    // modified by jishnu
  // cout << "inside cluster_expanded_energy \n";
  //basiplet.get_index();  //commented by jishnu
  string filename_eciout = "eci.out";  // added by jishnu
  ifstream eciout; // added by jishnu
  eciout.open(filename_eciout.c_str()); // added by jishnu
  basiplet.read_eci(eciout);  // added by jishnu

  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      //test that the number of correlations matches the number of clusters in basiplet
      if(basiplet.size.size() != superstruc[sc].conf[nc].correlations.size()){
	cout << "Cannot calculate cluster expanded energy since the correlations and clusters are not compatible\n";
	return;
      }
      superstruc[sc].conf[nc].cefenergy=0.0;
      for(int i=0; i<basiplet.size.size(); i++){
	int s=basiplet.size[i];
	int o=basiplet.order[i];
	superstruc[sc].conf[nc].cefenergy=superstruc[sc].conf[nc].cefenergy+basiplet.orb[s][o].equiv.size()*basiplet.orb[s][o].eci*superstruc[sc].conf[nc].correlations[i];
      }
      superstruc[sc].conf[nc].got_cefenergy = true;
    }
  }
  return;
}

//************************************************************
// ***********************************************************
void configurations::CEfenergy_analysis(){  // added by jishnu

  cluster_expanded_energy();
  int dim = chull.face[0].corner.size()-1;

  int ssc,nnc;
  for(int i=0;i<chull.point.size();i++){
    bool namematch = false;
    for(int sc=0; sc<superstruc.size(); sc++){
      for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
	if(chull.point[i].name == superstruc[sc].conf[nc].name) {namematch = true;ssc = sc; nnc = nc; break;}
      }
      if(namematch == true) break;
    }
    chull.point[i].fpfenergy = superstruc[ssc].conf[nnc].fpfenergy;
    chull.point[i].cefenergy = superstruc[ssc].conf[nnc].cefenergy;
    chull.point[i].update_ce();
  }

  for(int i=0;i<chull.face.size(); i++){
    for(int j=0;j<chull.face[i].corner.size();j++){
      bool namematch = false;
      for(int sc=0; sc<superstruc.size(); sc++){
	for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
	  if(chull.face[i].corner[j].name == superstruc[sc].conf[nc].name) {namematch = true;ssc = sc; nnc = nc; break;}
	}
	if(namematch == true) break;
      }

      chull.face[i].corner[j].fpfenergy = superstruc[ssc].conf[nnc].fpfenergy;
      chull.face[i].corner[j].cefenergy = superstruc[ssc].conf[nnc].cefenergy;
      chull.face[i].corner[j].update_ce();
    }
  }

  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      superstruc[sc].conf[nc].update_ce();
    }
  }

  // ------------- this writes the vasp hull with clex energy--------
  string FP_hull_clex_en_file="FPhull.clex";
  ofstream FPhullclexen;
  FPhullclexen.open(FP_hull_clex_en_file.c_str());
  if(!FPhullclexen){
    cout << "cannot open FPhull.clex file.\n";
    return;
  }
  FPhullclexen << "# clex_form_en        FP_form_en         concentrations             meaningless           name \n";
  for(int i=0; i<chull.point.size(); i++){
    FPhullclexen <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << chull.point[i].coordinate[chull.point[i].coordinate.size()-1];
    FPhullclexen <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << chull.point[i].fpfenergy;
    for(int j=0;j<(chull.point[i].coordinate.size()-1);j++){
      FPhullclexen <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) <<  chull.point[i].coordinate[j];
    }
    FPhullclexen <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << 0;
    FPhullclexen <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(15)<<setprecision(0) << chull.point[i].name << "\n";
  }

  FPhullclexen.close();

  //----------------------------------------------------------------

  get_delE_from_hull_w_clexen();

  string energyclex = "energy.clex";
  ofstream enclex;
  enclex.open(energyclex.c_str());
  if(!enclex){
    cout << "cannot open energy.clex file \n";
    return;
  }

  string belowhull_file = "below.hull";
  ofstream belowhull;
  belowhull.open(belowhull_file.c_str());
  if(!belowhull){
    cout << "cannot open below.hull file \n";
    return;
  }

  enclex << "formation energy           calculated/not          concentrations            dist_from_hull            name \n";

  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      enclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << superstruc[sc].conf[nc].cefenergy;
      if(superstruc[sc].conf[nc].calculated)enclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << 1;
      else enclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << 0;
      for(int j=0;j<dim;j++){
	enclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) <<  superstruc[sc].conf[nc].coordinate[j];
      }
      enclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << superstruc[sc].conf[nc].delE_from_facet;
      enclex <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << superstruc[sc].conf[nc].name;
      enclex << "\n";
    }
  }
  enclex.close();

  belowhull << "formation energy           calculated/not          concentrations            dist_from_hull            name \n";
  int number_belowhull = 0;

  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(superstruc[sc].conf[nc].delE_from_facet < (-tol)){
	if (superstruc[sc].conf[nc].calculated) {
	  number_belowhull++;
	  belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << superstruc[sc].conf[nc].cefenergy;
	  belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << 1;
	  for(int j=0;j<dim;j++){
	    belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) <<  superstruc[sc].conf[nc].coordinate[j];
	  }
	  belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << superstruc[sc].conf[nc].delE_from_facet;
	  belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << superstruc[sc].conf[nc].name;
	  belowhull << "\n";
        }
      }
    }
  }

  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(superstruc[sc].conf[nc].delE_from_facet < (-tol)){
	if (!superstruc[sc].conf[nc].calculated) {
	  number_belowhull++;
	  belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << superstruc[sc].conf[nc].cefenergy;
	  belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << 0;
	  for(int j=0;j<dim;j++){
	    belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) <<  superstruc[sc].conf[nc].coordinate[j];
	  }
	  belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << superstruc[sc].conf[nc].delE_from_facet;
	  belowhull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << superstruc[sc].conf[nc].name;
	  belowhull << "\n";
        }
      }
    }
  }

  belowhull << "Total No of below Hull points = "  << number_belowhull << "\n";
  belowhull.close();

  chull.clear_arrays();
  assemble_hull();
  chull.write_clex_hull();

}
//************************************************************

void configurations::print_eci_inputfiles_old(){

  //first determine how many calculated structures there are
  int num_calc=0;
  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(superstruc[sc].conf[nc].calculated){
	num_calc++;
      }
    }
  }

  //determine how many basis clusters there are
  basiplet.get_hierarchy();

  //now print out the corr.in and ener.in files
  string corr_file="corr.in";
  string ener_master_file="ener_master.in";
  string ener_file="ener.in";
  string energyold_file="energy_old";

  ofstream corr;
  corr.open(corr_file.c_str());
  if(!corr){
    cout << "cannot open corr.in file.\n";
    return;
  }


  ofstream ener;
  ener.open(ener_file.c_str());
  if(!ener){
    cout << "cannot open ener.in file.\n";
    return;
  }

  ofstream ener_master;
  ener_master.open(ener_master_file.c_str());
  if(!ener_master){
    cout << "cannot open ener_master.in file.\n";
    return;
  }

  ofstream energyold;
  energyold.open(energyold_file.c_str());
  if(!energyold){
    cout << "cannot open energy_old file.\n";
    return;
  }


  corr << basiplet.size.size() << " # number of clusters\n";
  corr << num_calc << " # number of configurations\n";
  corr << "clusters \n";

  ener_master << "       exact_ener    weight   structure name  \n";

  ener << "       exact_ener    weight   structure name  \n";

  energyold << "#   form_energy			weight			concentrations			dist_from_hull          name\n";

  num_calc=0;
  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(superstruc[sc].conf[nc].calculated){
	num_calc++;
	superstruc[sc].conf[nc].print_correlations(corr);
	ener_master << num_calc << "  " << superstruc[sc].conf[nc].fenergy << "   " << superstruc[sc].conf[nc].weight
		    << "    " << superstruc[sc].conf[nc].name << "\n";
	ener << num_calc << "  " << superstruc[sc].conf[nc].fenergy << "   " << superstruc[sc].conf[nc].weight
	     << "    " << superstruc[sc].conf[nc].name << "\n";
	superstruc[sc].conf[nc].print_in_energy_file(energyold);
      }
    }
  }

  corr.close();
  ener_master.close();
  ener.close();
  energyold.close();


}




//************************************************************

void configurations::print_eci_inputfiles(){   //changed by jishnu (ener.in and energy files are in energy now) (corr.in is in energy.corr now)

  //first determine how many calculated structures there are
  int num_calc=0;
  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(superstruc[sc].conf[nc].calculated){
	num_calc++;
      }
    }
  }

  //determine how many basis clusters there are
  basiplet.get_hierarchy();

  //now print out the energy.corr and enery files
  string corr_file="corr.in";
  string energy_file="energy";

  ofstream corr;
  corr.open(corr_file.c_str());
  if(!corr){
    cout << "cannot open corr.in file.\n";
    return;
  }

  ofstream energy;
  energy.open(energy_file.c_str());
  if(!energy){
    cout << "cannot open energy file.\n";
    return;
  }


  corr << basiplet.size.size() << " # number of clusters\n";
  corr << num_calc << " # number of configurations\n";
  corr << "clusters \n";

  energy << "# formation energy   weight        concentrations         dist_from_hull        name \n";
  get_delE_from_hull();
  num_calc=0;
  for(int sc=0; sc<superstruc.size(); sc++){
    for(int nc=0; nc<superstruc[sc].conf.size(); nc++){
      if(superstruc[sc].conf[nc].calculated){
	num_calc++;
	// corr << superstruc[sc].conf[nc].name;
	superstruc[sc].conf[nc].print_correlations(corr);
	//energy << num_calc << "      ";
	//energy << superstruc[sc].conf[nc].relax_step << "        ";
	//energy << superstruc[sc].conf[nc].weight << "       ";
	superstruc[sc].conf[nc].print_in_energy_file(energy);
      }
    }
  }

  corr.close();
  energy.close();

}



//************************************************************

void configurations::assemble_coordinate_fenergy(){
  for(int ns=0; ns<superstruc.size(); ns++){
    for(int nc=0; nc<superstruc[ns].conf.size(); nc++){
      superstruc[ns].conf[nc].assemble_coordinate_fenergy();
    }
  }
}




//************************************************************



//************************************************************
void facet::get_norm_vec(){  // added by jishnu  // finding a-b-c of ax+by+cz+d=0; // d is the offset

  normal_vec.clear();
  int num_row = corner.size() -1 ;
  int num_col = corner.size() -1 ;
  double vec_mag = 0.0;
  offset = 0.0;
  for(int i=0; i<corner.size(); i++){
    Array det_mat(num_row,num_col);
    double *matrix = new double [num_row*num_col];
    int mat_ind =0;

    // begin filling matrix
    for(int j=1;j<corner.size();j++){
      for(int k=0; k<corner[j].coordinate.size();k++){
	if(k!=i){
	  matrix[mat_ind]=corner[j].coordinate[k]-corner[0].coordinate[k];
	  mat_ind++;
	}
      }
    }
    det_mat.setArray(matrix);
    // matrix filled; find determinant

    double det_val=det_mat.det();
    det_val*=pow(-1.0,i+2.0);
    normal_vec.push_back(det_val);
    offset-=corner[0].coordinate[i]*det_val;
    vec_mag+=det_val*det_val;
  }

  vec_mag=pow(vec_mag,0.5);
  for(int i=0;i<normal_vec.size();i++){
    normal_vec[i]/=vec_mag;
  }

  offset/=vec_mag;


} // end of get_norm_vec


//************************************************************
//************************************************************
bool facet::find_endiff(arrangement arr, double &delE_from_facet) { // to find which facet contains which structure and to find the corresponding energy on the facet // added by jishnu

  vector <double> phase_frac;
  int dim_whole = corner.size();
  int dim = dim_whole - 1;
  vector <double> trow;
  vector <vector <double> > coord;
  vector <double> con_vec;

  for (int i=0;i<dim;i++){
    for (int j=0;j<dim;j++){
      trow.push_back(corner[i].coordinate[j]-corner[dim].coordinate[j]);
    }
    coord.push_back(trow);
    trow.clear();
  }

  for (int i=0;i<dim;i++){
    con_vec.push_back(arr.coordinate[i]-corner[dim].coordinate[i]);
  }

  double *ccoord = new double [dim*dim];
  for (int i=0;i<dim;i++){
    for (int j=0;j<dim;j++){
      ccoord[i*dim+j] = coord[i][j];
    }
  }

  double *ccon_vec = new double [dim];
  for (int i=0;i<dim;i++){
    ccon_vec[i] = con_vec[i];
  }

  Array cccoord(dim,dim);
  cccoord.setArray(ccoord);
  Array tr_cccoord = cccoord.transpose();
  Array inv_tr_cccoord = tr_cccoord.inverse();

  Array cccon_vec(dim);
  cccon_vec.setArray(ccon_vec);
  Array tr_cccon_vec = cccon_vec.transpose();

  Array pphase_frac = inv_tr_cccoord * tr_cccon_vec; // this is a column vector

  double sum = 0.0;

  for (int i=0;i<dim;i++){
    phase_frac.push_back(pphase_frac.elem(i+1,1));
    sum = sum + pphase_frac.elem(i+1,1);
  }
  phase_frac.push_back(1.0-sum);

  sum = 0.0;
  for (int i=0;i<dim;i++){
    sum = sum + normal_vec[i]*arr.coordinate[i];
  }

  // en_facet = ( -d - ax - by )/c;
  double en_facet = (-offset - sum)/normal_vec[dim];
  delE_from_facet = arr.fenergy - en_facet;
  /*// norm_dist_from_facet = mod(ax1+by1+cz1+d)/sqrt(a^2+b^2+c^2);
    double sum1 = 0.0;
    for(int i=0;i<dim_whole;i++){
    sum1+=normal_vec[i]*arr.coordinate[i];
    }
    sum1+=offset;
    double sum2 =0.0;
    for(int i=0;i<dim_whole;i++){
    sum1+=normal_vec[i]*normal_vec[i];
    }
    norm_dist_from_facet = fabs(sum1)/sqrt(sum2);	*/


  for(int i=0;i<dim_whole;i++){
    if((phase_frac[i] < 0.0) || (phase_frac[i] > 1.0)) return(false);
  }
  return (true);

}  // end of find_endiff
//************************************************************

//************************************************************
void facet::get_mu(){   // added by jishnu

  int no_of_comp = corner.size()-1;
  for (int i=0;i<no_of_comp;i++){
    double value = (-offset - normal_vec[i])/normal_vec[corner.size()];
    mu.push_back(value);
  }

} // end of get_mu
//************************************************************
//************************************************************
void hull::sort_conc(){

  if(point.size() == 0){
    cout << "no points in hull object \n";
    cout << "quitting sort_conc \n";
    return;
  }

  //sort the last column first and go left in the coordinate vector of point[]

  for(int c=point[0].coordinate.size()-2; c >= 0; c--){
    for(int i=0; i<point.size(); i++){
      for(int j=i+1; j<point.size(); j++){
	if(point[i].coordinate[c] > point[j].coordinate[c]){
	  arrangement tarrange = point[j];
	  point[j]=point[i];
	  point[i]=tarrange;
	}
      }
    }
  }


}
//************************************************************
void hull::write_hull(){

  string hull_file="hull";
  ofstream hull;
  hull.open(hull_file.c_str());
  if(!hull){
    cout << "cannot open hull file.\n";
    return;
  }
  hull << "# formation_energy        meaningless          concentrations            meaningless            name \n";
  for(int i=0; i<point.size(); i++){
    hull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << point[i].coordinate[point[i].coordinate.size()-1];
    hull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) << 0;
    for(int j=0;j<(point[i].coordinate.size()-1);j++){
      hull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(20)<<setprecision(9) <<  point[i].coordinate[j];
    }
    hull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(15)<<setprecision(0) << 0;
    hull <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(15)<<setprecision(0) << point[i].name << "\n";
  }

  hull.close();


  string facet_file="facet";
  ofstream facett;
  facett.open(facet_file.c_str());
  if(!facett){
    cout << "cannot open facet file.\n";
    return;
  }

  for(int i=0;i<face.size();i++){
    // cout << face[i].normal_vec[0] << "   "<< face[i].normal_vec[1]<< "     "<< face[i].offset << "  \n";
    for(int j=0;j<face[i].corner.size();j++){
      facett <<setiosflags(ios::right)<<setiosflags(ios::fixed)<<setw(15)<<setprecision(0)<< face[i].corner[j].name;
    }
    facett << "\n";
  }

  facett.close();


}
//************************************************************

//************************************************************

void chempot::initialize(concentration conc){
  for(int i=0; i<conc.compon.size(); i++){
    vector<double> tm;
    vector<specie> tcompon;
    for(int j=0; j<conc.compon[i].size(); j++){
      tm.push_back(0);
      tcompon.push_back(conc.compon[i][j]);
    }
    m.push_back(tm);
    compon.push_back(tcompon);
  }
}
//************************************************************

void chempot::initialize(vector<vector< specie > > init_compon){
  for(int i=0; i<init_compon.size(); i++){
    vector<double> tm;
    vector<specie> tcompon;
    for(int j=0; j<init_compon[i].size(); j++){
      tm.push_back(0);
      tcompon.push_back(init_compon[i][j]);
    }
    m.push_back(tm);
    compon.push_back(tcompon);
  }
}



//************************************************************

void chempot::set(facet face){
  int k=0;
  for(int i=0; i<m.size(); i++){
    for(int j=0; j<m[i].size()-1; j++){
      m[i][j]=face.mu[k];
      k++;
    }
    m[i][m[i].size()-1]=0.0;
  }
}


//************************************************************

void chempot::increment(chempot muinc){
  if(m.size() !=muinc.m.size()){
    cout << "Trying to increment chemical potential with wrong dimensioned increment \n";
    return;
  }
  for(int i=0; i<m.size(); i++){
    if(m[i].size() !=muinc.m[i].size()){
      cout << "Trying to increment chemical potential with wrong dimensioned increment \n";
      return;
    }

    for(int j=0; j<m[i].size(); j++){
      m[i][j]=m[i][j]+muinc.m[i][j];
    }
  }

}


//************************************************************

void chempot::print(ostream &stream){
  for(int i=0; i<m.size(); i++){
    for(int j=0; j<m[i].size(); j++){
      stream << m[i][j] << "  ";
    }
  }
}


//************************************************************

void chempot::print_compon(ostream &stream){
  for(int i=0; i<compon.size(); i++){
    for(int j=0; j<compon[i].size(); j++){
      compon[i][j].print(stream);
      stream << "  ";
    }
  }
}




//************************************************************

void trajectory::initialize(concentration conc){
  Rx.clear();
  Ry.clear();
  Rz.clear();
  R2.clear();
  spin.clear();
  for(int i=0; i<conc.compon.size(); i++){
    vector<double> tR;
    vector<int> tspin;
    vector<specie> telements;
    for(int j=0; j<conc.compon[i].size(); j++){
      tR.push_back(0.0);
      tspin.push_back(conc.compon[i][j].spin);
      telements.push_back(conc.compon[i][j]);
    }
    Rx.push_back(tR);
    Ry.push_back(tR);
    Rz.push_back(tR);
    R2.push_back(tR);
    spin.push_back(tspin);
    elements.push_back(telements);
  }
}


//************************************************************

void trajectory::set_zero(){
  for(int i=0; i<Rx.size(); i++){
    for(int j=0; j<Rx[i].size(); j++){
      Rx[i][j]=0.0;
      Ry[i][j]=0.0;
      Rz[i][j]=0.0;
      R2[i][j]=0.0;
    }
  }


}


//************************************************************

void trajectory::increment(trajectory R){

  if(Rx.size() != R.Rx.size()){
    cout << "incompatibility in trajectory incrementer \n";
    return;
  }

  for(int i=0; i<Rx.size(); i++){

    if(Rx[i].size() != R.Rx[i].size()){
      cout << "incompatibility in trajectory incrementer \n";
      return;
    }

    for(int j=0; j<Rx[i].size(); j++){
      Rx[i][j]=Rx[i][j]+R.Rx[i][j];
      Ry[i][j]=Ry[i][j]+R.Ry[i][j];
      Rz[i][j]=Rz[i][j]+R.Rz[i][j];
      R2[i][j]=R2[i][j]+R.R2[i][j];
    }
  }
}


//************************************************************

void trajectory::normalize(double D){
  for(int i=0; i<spin.size(); i++){
    for(int j=0; j<spin[i].size(); j++){
      R2[i][j]=R2[i][j]/D;
    }
  }
}


//************************************************************

void trajectory::normalize(concentration conc){
  for(int i=0; i<spin.size(); i++){
    for(int j=0; j<spin[i].size(); j++){
      if(abs(conc.occup[i][j]) > tol){
	R2[i][j]=R2[i][j]/conc.occup[i][j];
      }
    }
  }
}


//************************************************************

void trajectory::print(ostream &stream){
  for(int i=0; i<R2.size(); i++){
    for(int j=0; j<R2[i].size(); j++){
      stream << R2[i][j] << "  ";
    }
  }
}



//************************************************************

void trajectory::print_elements(ostream &stream){
  for(int i=0; i<elements.size(); i++){
    for(int j=0; j<elements[i].size(); j++){
      elements[i][j].print(stream);
      stream << "  ";
    }
  }
}



//************************************************************

Monte_Carlo::Monte_Carlo(structure in_prim, structure in_struc, multiplet in_basiplet, int idim, int jdim, int kdim){

  prim=in_prim;
  basiplet=in_basiplet;
  di=idim; dj=jdim; dk=kdim;

  prim.get_trans_mat();
  prim.update_lat();
  prim.update_struc();

  //check whether the monte carlo cell dimensions are compatible with those of init_struc
  //if not, use the new ones suggested by compatible

  int ndi,ndj,ndk;
  if(!compatible(in_struc,ndi,ndj,ndk)){
    di=ndi; dj=ndj; dk=ndk;
    cout << "New Monte Carlo cell dimensions have been chosen to make the cell\n";
    cout << "commensurate with the initial configuration.\n";
    cout << "The dimensions now are " << di << " " << dj << " " << dk << "\n";
    cout << "\n";
  }

  nuc=di*dj*dk;

  si=6*di; sj=6*dj; sk=6*dk;
  arrayi = new int [2*di];
  arrayj = new int [2*dj];
  arrayk = new int [2*dk];
  for(int i=0; i<di; ++i){arrayi[i] = i; arrayi[i+di] = i;}
  for(int j=0; j<dj; ++j){arrayj[j] = j; arrayj[j+dj] = j;}
  for(int k=0; k<dk; ++k){arrayk[k] = k; arrayk[k+dk] = k;}

  idum=time(NULL);

  //collect basis sites and determine # of basis sites bd
  collect_basis();
  db=basis.size();



  ind1=di*dj*dk; ind2=dj*dk; ind3=dk;
  nmcL=di*dj*dk*db;
  mcL= new int[nmcL];
  ltoi = new int[nmcL];
  ltoj = new int[nmcL];
  ltok = new int[nmcL];
  ltob = new int[nmcL];

  for(int i=0; i<di; i++){
    for(int j=0; j<dj; j++){
      for(int k=0; k<dk; k++){
	for(int b=0; b<db; b++){
	  int l=index(i,j,k,b);
	  ltoi[l]=i;
	  ltoj[l]=j;
	  ltok[l]=k;
	  ltob[l]=b;
	}
      }
    }
  }

  //initialize the concentration variables
  conc.collect_components(prim);
  collect_sublat();
  num_atoms=conc;
  num_hops=conc;
  AVconc=conc;
  AVsublat_conc=sublat_conc;
  AVnum_atoms=num_atoms;
  assemble_conc_basis_links();

  //initialize the susceptibility and thermodynamic factor variables
  Susc.initialize(conc);
  AVSusc.initialize(conc);

  cout << "Dimensions of Susc are = " << Susc.f.size() << " and " << Susc.f[0].size() << "\n";

  //initialize correlation vector

  for(int i=0; i<basiplet.orb.size(); i++){
    for(int j=0; j<basiplet.orb[i].size(); j++){
      AVcorr.push_back(0.0);
    }
  }

  cout << "AVcorr initialized with " << AVcorr.size() << " elements.\n";

  //initialize the occupation variables in mcL array with the arrangement in init_struc
  initialize(in_struc);

  generate_ext_monteclust(basis, basiplet, montiplet);
  generate_eci_arrays();
}




//************************************************************
void Monte_Carlo::collect_basis(){
  for(int na=0; na<prim.atom.size(); na++){
    if(prim.atom[na].compon.size() >= 2) basis.push_back(prim.atom[na]);
  }
  for(int nb=0; nb<basis.size(); nb++){
    basis[nb].assign_spin();
  }
}


//************************************************************

void Monte_Carlo::assemble_conc_basis_links(){

  //first make the basis to concentration vector basis_to_conc
  for(int i=0; i<basis.size(); i++){
    basis_to_conc.push_back(-1);
    for(int j=0; j<conc.compon.size(); j++){
      if(compare(basis[i].compon,conc.compon[j]))basis_to_conc[i]=j;
    }
    if(basis_to_conc[i] == -1){
      cout << "incompatibility between the basis and the concentration object\n";
      cout << "quitting assemble_conc_links() \n";
      return;
    }
  }

  //next make the conc_to_basis vector
  for(int i=0; i<conc.compon.size(); i++){
    vector<int> tconc_to_basis;
    for(int j=0; j<basis.size(); j++){
      if(compare(basis[j].compon,conc.compon[i])) tconc_to_basis.push_back(j);
    }
    if(tconc_to_basis.size() == 0){
      cout << "incompatibility between the basis and the concentration object\n";
      cout << "quitting assemble_conc_links() \n";
      return;
    }
    conc_to_basis.push_back(tconc_to_basis);
  }
}





//************************************************************
void Monte_Carlo::collect_sublat(){

  //first generate the orbits of non equivalent points
  vector<orbit> points;

  for(int i=0; i< basis.size(); i++){
    //check whether the basis site already exists in an orbit
    bool found = false;
    ////////////////////////////////////////////////////////////////////////////////
    //cout << "points size: " << points.size() << "\n";
    ////////////////////////////////////////////////////////////////////////////////
    for(int np=0; np<points.size(); np++){
      for(int ne=0; ne<points[np].equiv.size(); ne++){
	////////////////////////////////////////////////////////////////////////////////
	//swoboda
	//cout << "\ncompare basis and points\n";
	//for(int x=0; x<3; x++){
	//    cout << "basis[" << i << "]: " << basis[i].fcoord[x] << "\tp[" << np << "]equiv[" << ne << "]: " <<
	//    points[np].equiv[ne].point[0].fcoord[x] << "\n";
	//}
	//cout << "\n";
	////////////////////////////////////////////////////////////////////////////////
	if(compare(basis[i],points[np].equiv[ne].point[0])) found = true;
      }
    }
    if(!found){
      cluster tclust;
      tclust.point.push_back(basis[i]);
      orbit torb;
      torb.equiv.push_back(tclust);
      get_equiv(torb,prim.factor_group);
      points.push_back(torb);
    }
  }


  cout << " THE NUMBER OF DISTINCT POINT CLUSTERS ARE \n";
  cout << points.size() << "\n";



  for(int i=0; i<points.size(); i++){
    sublat_conc.compon.push_back(points[i].equiv[0].point[0].compon);
    vector< double> toccup;
    for(int j=0; j<points[i].equiv[0].point[0].compon.size(); j++) toccup.push_back(0.0);
    sublat_conc.occup.push_back(toccup);
    sublat_conc.mu.push_back(toccup);
  }

  // fill the basis_to_sublat vector

  for(int i=0; i<basis.size(); i++){
    bool mapped= false;
    for(int j=0; j<points.size(); j++){
      for(int k=0; k< points[j].equiv.size(); k++){
	if(compare(basis[i],points[j].equiv[k].point[0])){
	  mapped = true;
	  basis_to_sublat.push_back(j);
	}
      }
    }
    if(!mapped){
      cout << " unable to map a basis site to a sublattice site \n";
    }
  }

  // fill the sublat_to_basis double vector

  for(int j=0; j<points.size(); j++){
    vector<int> tbasis;
    for(int i=0; i<basis.size(); i++){
      if(basis_to_sublat[i] == j) tbasis.push_back(i);
    }
    sublat_to_basis.push_back(tbasis);
  }

}





//************************************************************

void Monte_Carlo::update_mu(chempot mu){
  //takes the mu's and puts them in the right spots in basis
  for(int i=0; i<basis.size(); i++){
    basis[i].mu.clear();
    for(int j=0; j<basis[i].compon.size(); j++){
      basis[i].mu.push_back(mu.m[basis_to_conc[i]][j]);
    }
    basis[i].assemble_flip();
  }

}




//************************************************************
void Monte_Carlo::invert_index(int &i, int &j, int &k, int &b, int l){
  k = l % dk;
  j = ((l - k) / dk) % dj;
  i = ((l - k - j*dk) / (dj*dk)) % di ;
  b = (l - k - j*dk - i*dj*dk) / (dk*dj*di) ;
}

//************************************************************

void Monte_Carlo::generate_eci_arrays(){

  //arrays that need to be constructed:
  //     - eci
  //     - multiplicity
  //     - number of points
  //     - shift array
  //     - start_end array (contains initial and final indices for the different bases in all above arrays)

  //first work out the dimensions so that arrays with appropriate lengths can be dimensioned
  //     for each basis site, go through the clusters and find those with non-zero eci - count the number of them
  //     for each basis site, go through all sites and exponents of the clusters and count the number of them

  neci=0;
  ns=0;
  nse=4*db;

  // determine neci and ns

  for(int nm=0; nm<montiplet.size(); nm++){
    for(int np=1; np<montiplet[nm].orb.size(); np++){
      for(int no=0; no<montiplet[nm].orb[np].size(); no++){
        if(abs(montiplet[nm].orb[np][no].eci) >= 0.000001){
          neci++;
          for(int ne=0; ne<montiplet[nm].orb[np][no].equiv.size(); ne++){
            for(int n=0; n<montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){
              ns=ns+(montiplet[nm].orb[np][no].equiv[ne].point[n].bit+1)*4;
            }
          }
        }
      }
    }
  }
  //allocate memory for s eci, mult, nump and startend ( determined by number of basis sites)

  s= new int[ns];
  eci= new double[neci];
  nums= new int[neci];
  nump= new int[neci];
  mult= new int[neci];
  startend= new int[nse];

  //fill up all the arrays by going through them again

  //first assign the empty eci

  if(montiplet.size()>=1){
    if(montiplet[0].orb.size()>=1){
      if(montiplet[0].orb[0].size()>=1){
        eci_empty=montiplet[0].orb[0][0].eci;
      }
    }
  }
  //cout<<"empty eci is assigned \n";

  int i=0;
  int j=0;
  for(int nm=0; nm<montiplet.size(); nm++){
    startend[nm*4+0]=i;
    startend[nm*4+2]=j;
    for(int np=1; np<montiplet[nm].orb.size(); np++){
      for(int no=0; no<montiplet[nm].orb[np].size(); no++){
        if(fabs(montiplet[nm].orb[np][no].eci) >= 0.000001){
          eci[i]=montiplet[nm].orb[np][no].eci;
          mult[i]=montiplet[nm].orb[np][no].equiv.size();
          nums[i]=montiplet[nm].orb[np][no].equiv[0].point.size();
          if(nums[i] == 0){
            cout << "Serious problem: a cluster has zero points\n";
            cout << "There will be problems when calculating the energy \n";
          }
          nump[i]=0;
          for(int n=0; n<montiplet[nm].orb[np][no].equiv[0].point.size(); n++){
            nump[i]=nump[i]+montiplet[nm].orb[np][no].equiv[0].point[n].bit+1;
          }
          i++;

          for(int ne=0; ne<montiplet[nm].orb[np][no].equiv.size(); ne++){
            for(int n=0; n<montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){
              for(int nn=0; nn<montiplet[nm].orb[np][no].equiv[ne].point[n].bit+1; nn++){
                for(int nnn=0; nnn<4; nnn++){
                  s[j]=montiplet[nm].orb[np][no].equiv[ne].point[n].shift[nnn];
                  j++;
                }
              }
            }
          }

        }
      }
    }
    startend[nm*4+1]=i;
    startend[nm*4+3]=j;
  }
}


//************************************************************

void Monte_Carlo::write_point_energy(ofstream &out){

  // for each basis site, get all points that are accessed

  ////////////////////////////////////////////////////////////////////////////////
  //swoboda
  vector< vector< vector< mc_index > > > collect;

  for(int nm=0; nm< montiplet.size(); nm++){
    vector< vector< mc_index > > tcollect;
    for(int np=1; np<montiplet[nm].orb.size(); np++){
      for(int no=0; no<montiplet[nm].orb[np].size(); no++){
	if(abs(montiplet[nm].orb[np][no].eci) >= 0.000001){
	  for(int ne=0; ne<montiplet[nm].orb[np][no].equiv.size(); ne++){
	    for(int n=0; n<montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){
	      vector <mc_index> tempcollect;
	      mc_index tpoint;
	      tpoint.basis_flag=montiplet[nm].orb[np][no].equiv[ne].point[n].basis_flag;
	      for(int i=0; i<4; i++){
		tpoint.shift[i]=montiplet[nm].orb[np][no].equiv[ne].point[n].shift[i];
		tpoint.num_specie=montiplet[nm].orb[np][no].equiv[ne].point[n].compon.size();
	      }
	      bool already_present=false;
	      for(int i=0; i<tcollect.size(); i++){
		if(compare(tpoint,tcollect[i][0])){
		  already_present=true;
		  //adding check to ensure size of collect[] is correct
		  if(tcollect[i].size() < (montiplet[nm].orb[np][no].equiv[ne].point[n].bit+1)){
		    for(int j=tcollect[i].size(); j<montiplet[nm].orb[np][no].equiv[ne].point[n].bit+1; j++){
		      mc_index tempoint;
		      tempoint=tpoint;
		      string j_num;
		      string p_num;
		      tempoint.name="p";
		      int_to_string(i,p_num,10);
		      tempoint.name.append(p_num);
		      int_to_string(j,j_num,10);
		      tempoint.name.append(j_num);
		      tcollect[i].push_back(tempoint);
		    }
		  }
		  break;
		}
	      }
	      if(already_present == false){
		tpoint.l=tcollect.size();
		string p_num;
		int_to_string(tpoint.l,p_num,10);
		tpoint.name="p";
		tpoint.name.append(p_num);
		for(int j=0; j<montiplet[nm].orb[np][no].equiv[ne].point[n].bit+1; j++){
		  mc_index tempoint;
		  tempoint=tpoint;
		  string j_num;
		  int_to_string(j,j_num,10);
		  tempoint.name.append(j_num);
		  tempcollect.push_back(tempoint);
		}
		tcollect.push_back(tempcollect);
	      }
	    }
	  }
	}
      }
    }
    collect.push_back(tcollect);
  }

  ////////////////////////////////////////////////////////////////////////////////

  out << "double Monte_Carlo::pointenergy(int i, int j, int k, int b){\n";

  out << "  double energy = 0.0;\n";
  //  if(montiplet.size() > 0){
  //    if(montiplet[0].orb.size() > 0){
  //      if(montiplet[0].orb[0].size() > 0) out << montiplet[0].orb[0][0].eci << ";\n";
  //      else out << "0.0;\n";
  //    }
  //    else out << "0.0;\n";
  //  }
  //  else out << "0.0;\n";



  out << "  int l; " << "\n";

  ////////////////////////////////////////////////////////////////////////////////
  //swoboda
  for(int b=0; b<collect.size(); b++){
    out << "  if(b == " << b << "){\n";

    for(int n=0; n<collect[b].size(); n++){
      for(int m=0; m<collect[b][n].size(); m++){
	if(collect[b][n][m].basis_flag=='1'){
	  //for occ basis
	  if(collect[b][n][m].num_specie%2!=0){
	    int num;
	    num=(collect[b][n][m].num_specie-1)/2;
	    out << "     l=index(i";
	    if(collect[b][n][m].shift[0] == 0) out << ",j";
	    if(collect[b][n][m].shift[0] < 0) out << collect[b][n][m].shift[0] << ",j";
	    if(collect[b][n][m].shift[0] > 0) out << "+" << collect[b][n][m].shift[0] << ",j";

	    if(collect[b][n][m].shift[1] == 0) out << ",k";
	    if(collect[b][n][m].shift[1] < 0) out << collect[b][n][m].shift[1] << ",k";
	    if(collect[b][n][m].shift[1] > 0) out << "+" << collect[b][n][m].shift[1] << ",k";

	    if(collect[b][n][m].shift[2] == 0) out << ",";
	    if(collect[b][n][m].shift[2] < 0) out << collect[b][n][m].shift[2] << ",";
	    if(collect[b][n][m].shift[2] > 0) out << "+" << collect[b][n][m].shift[2] << ",";

	    out << collect[b][n][m].shift[3] << ");\n";

	    string n_num,m_num;
	    string n_name,t_name;
	    int_to_string(n,n_num,10);
	    n_name="num";
	    t_name="t";
	    n_name.append(n_num);
	    t_name.append(n_num);
	    int_to_string(m,m_num,10);
	    n_name.append(m_num);
	    t_name.append(m_num);

	    //	    out << "     int " << t_name << "=" << m+1 << ";\n";
	    //	    out << "     int " << n_name << "=" << num << ";\n";
	    //                      out << "     int " << collect[b][n][m].name << "=floor(1/(abs(mcL[l]-" <<
	    //			n_name << "+" << t_name << "-1)+0.9));\n";
	    out << "     double " << collect[b][n][m].name << "= 0.5*mcL[l]+0.5;\n";
	  }
	  else{
	    int num;
	    num=collect[b][n][m].num_specie/2;
	    out << "     l=index(i";
	    if(collect[b][n][m].shift[0] == 0) out << ",j";
	    if(collect[b][n][m].shift[0] < 0) out << collect[b][n][m].shift[0] << ",j";
	    if(collect[b][n][m].shift[0] > 0) out << "+" << collect[b][n][m].shift[0] << ",j";

	    if(collect[b][n][m].shift[1] == 0) out << ",k";
	    if(collect[b][n][m].shift[1] < 0) out << collect[b][n][m].shift[1] << ",k";
	    if(collect[b][n][m].shift[1] > 0) out << "+" << collect[b][n][m].shift[1] << ",k";

	    if(collect[b][n][m].shift[2] == 0) out << ",";
	    if(collect[b][n][m].shift[2] < 0) out << collect[b][n][m].shift[2] << ",";
	    if(collect[b][n][m].shift[2] > 0) out << "+" << collect[b][n][m].shift[2] << ",";

	    out << collect[b][n][m].shift[3] << ");\n";

	    string m_num,n_num;
	    string n_name,t_name;
	    int_to_string(n,n_num,10);
	    n_name="num";
	    t_name="t";
	    n_name.append(n_num);
	    t_name.append(n_num);
	    int_to_string(m,m_num,10);
	    n_name.append(m_num);
	    t_name.append(m_num);

	    //	    out << "     int " << t_name << "=" << m+1 << ";\n";
	    //	    out << "     int " << n_name << "=" << num << ";\n";
	    //	    if(m+1<=num) out << "     int " << collect[b][n][m].name << "=floor(1/(abs(mcL[l]-" <<
	    //			   n_name << "+" << t_name << "-1)+0.9));\n";
	    //	    else out << "     int " << collect[b][n][m].name << "=floor(1/(abs(mcL[l]-" <<
	    //		   n_name << "+" << t_name << ")+.9));\n";
	    out << "     double " << collect[b][n][m].name << "=0.5*mcL[l]+0.5;\n";
	  }
	}

	else{
	  //for spin basis
	  out << "     l=index(i";
	  if(collect[b][n][m].shift[0] == 0) out << ",j";
	  if(collect[b][n][m].shift[0] < 0) out << collect[b][n][m].shift[0] << ",j";
	  if(collect[b][n][m].shift[0] > 0) out << "+" << collect[b][n][m].shift[0] << ",j";

	  if(collect[b][n][m].shift[1] == 0) out << ",k";
	  if(collect[b][n][m].shift[1] < 0) out << collect[b][n][m].shift[1] << ",k";
	  if(collect[b][n][m].shift[1] > 0) out << "+" << collect[b][n][m].shift[1] << ",k";

	  if(collect[b][n][m].shift[2] == 0) out << ",";
	  if(collect[b][n][m].shift[2] < 0) out << collect[b][n][m].shift[2] << ",";
	  if(collect[b][n][m].shift[2] > 0) out << "+" << collect[b][n][m].shift[2] << ",";

	  out << collect[b][n][m].shift[3] << ");\n";
	  out << "     int " << collect[b][n][m].name << "=mcL[l]";
	  if(m==0) out << ";\n";
	  if(m>0){
	    for(int mm=0; mm<m; mm++){
	      out << "*mcL[l]";
	      if(mm==m-1) out << ";\n";
	    }
	  }
	}
      }
    }

    out << "\n";

    out << "     energy = energy";

    int nm=b;
    for(int np=1; np<montiplet[nm].orb.size(); np++){
      for(int no=0; no<montiplet[nm].orb[np].size(); no++){
	if(fabs(montiplet[nm].orb[np][no].eci) >= 0.000001){
	  if(montiplet[nm].orb[np][no].eci < 0.0) out << montiplet[nm].orb[np][no].eci << "*(";
	  if(montiplet[nm].orb[np][no].eci > 0.0) out << "+" << montiplet[nm].orb[np][no].eci << "*(";
	  for(int ne=0; ne<montiplet[nm].orb[np][no].equiv.size(); ne++){
	    if(ne > 0) out << "+";
	    for(int n=0; n<montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){
	      mc_index tpoint;
	      for(int i=0; i<4; i++){
		tpoint.shift[i]=montiplet[nm].orb[np][no].equiv[ne].point[n].shift[i];
	      }
	      for(int i=0; i<collect[b].size(); i++){
		if(compare(tpoint,collect[b][i][0])){
		  int j;
		  j=montiplet[nm].orb[np][no].equiv[ne].point[n].bit;
		  //cout << "b: " << b << "\ti: " << i << "\tj: " << j << "\tname: " << collect[b][i][j].name << "\n";
		  out << collect[b][i][j].name;
		  if(n == montiplet[nm].orb[np][no].equiv[ne].point.size()-1){
		    out << "";
		  }
		  else out << "*";
		  break;
		}
	      }
	    }
	  }
	  out << ")";
	}
      }
    }
    out << ";\n";
    out << "     return energy;\n";
    out << "  }\n";
    out << "\n";
    out << "\n";
  }

  out << "}\n";

}





//************************************************************

void Monte_Carlo::write_normalized_point_energy(ofstream &out){

  // for each basis site, get all points that are accessed

  ////////////////////////////////////////////////////////////////////////////////
  //swoboda
  vector< vector< vector< mc_index > > > collect;

  for(int nm=0; nm< montiplet.size(); nm++){
    vector< vector< mc_index > > tcollect;
    for(int np=1; np<montiplet[nm].orb.size(); np++){
      for(int no=0; no<montiplet[nm].orb[np].size(); no++){
	if(abs(montiplet[nm].orb[np][no].eci) >= 0.000001){
	  for(int ne=0; ne<montiplet[nm].orb[np][no].equiv.size(); ne++){
	    for(int n=0; n<montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){
	      vector <mc_index> tempcollect;
	      mc_index tpoint;
	      tpoint.basis_flag=montiplet[nm].orb[np][no].equiv[ne].point[n].basis_flag;
	      for(int i=0; i<4; i++){
		tpoint.shift[i]=montiplet[nm].orb[np][no].equiv[ne].point[n].shift[i];
		tpoint.num_specie=montiplet[nm].orb[np][no].equiv[ne].point[n].compon.size();
	      }
	      bool already_present=false;
	      for(int i=0; i<tcollect.size(); i++){
		if(compare(tpoint,tcollect[i][0])){
		  already_present=true;
		  //adding check to ensure size of collect[] is correct
		  if(tcollect[i].size() < (montiplet[nm].orb[np][no].equiv[ne].point[n].bit+1)){
		    for(int j=tcollect[i].size(); j<montiplet[nm].orb[np][no].equiv[ne].point[n].bit+1; j++){
		      mc_index tempoint;
		      tempoint=tpoint;
		      string j_num;
		      string p_num;
		      tempoint.name="p";
		      int_to_string(i,p_num,10);
		      tempoint.name.append(p_num);
		      int_to_string(j,j_num,10);
		      tempoint.name.append(j_num);
		      tcollect[i].push_back(tempoint);
		    }
		  }
		  break;
		}
	      }
	      if(already_present == false){
		tpoint.l=tcollect.size();
		string p_num;
		int_to_string(tpoint.l,p_num,10);
		tpoint.name="p";
		tpoint.name.append(p_num);
		for(int j=0; j<montiplet[nm].orb[np][no].equiv[ne].point[n].bit+1; j++){
		  mc_index tempoint;
		  tempoint=tpoint;
		  string j_num;
		  int_to_string(j,j_num,10);
		  tempoint.name.append(j_num);
		  tempcollect.push_back(tempoint);
		}
		tcollect.push_back(tempcollect);
	      }
	    }
	  }
	}
      }
    }
    collect.push_back(tcollect);
  }

  ////////////////////////////////////////////////////////////////////////////////

  out << "double Monte_Carlo::normalized_pointenergy(int i, int j, int k, int b){\n";

  out << "  double energy = 0.0;\n";
  //  if(montiplet.size() > 0){
  //    if(montiplet[0].orb.size() > 0){
  //      if(montiplet[0].orb[0].size() > 0) out << montiplet[0].orb[0][0].eci << ";\n";
  //      else out << "0.0;\n";
  //    }
  //    else out << "0.0;\n";
  //  }
  //  else out << "0.0;\n";



  out << "  int l; " << "\n";

  ////////////////////////////////////////////////////////////////////////////////
  //swoboda
  for(int b=0; b<collect.size(); b++){
    out << "  if(b == " << b << "){\n";

    for(int n=0; n<collect[b].size(); n++){
      for(int m=0; m<collect[b][n].size(); m++){
	if(collect[b][n][m].basis_flag=='1'){
	  //for occ basis
	  if(collect[b][n][m].num_specie%2!=0){
	    int num;
	    num=(collect[b][n][m].num_specie-1)/2;
	    out << "     l=index(i";
	    if(collect[b][n][m].shift[0] == 0) out << ",j";
	    if(collect[b][n][m].shift[0] < 0) out << collect[b][n][m].shift[0] << ",j";
	    if(collect[b][n][m].shift[0] > 0) out << "+" << collect[b][n][m].shift[0] << ",j";

	    if(collect[b][n][m].shift[1] == 0) out << ",k";
	    if(collect[b][n][m].shift[1] < 0) out << collect[b][n][m].shift[1] << ",k";
	    if(collect[b][n][m].shift[1] > 0) out << "+" << collect[b][n][m].shift[1] << ",k";

	    if(collect[b][n][m].shift[2] == 0) out << ",";
	    if(collect[b][n][m].shift[2] < 0) out << collect[b][n][m].shift[2] << ",";
	    if(collect[b][n][m].shift[2] > 0) out << "+" << collect[b][n][m].shift[2] << ",";

	    out << collect[b][n][m].shift[3] << ");\n";

	    string n_num,m_num;
	    string n_name,t_name;
	    int_to_string(n,n_num,10);
	    n_name="num";
	    t_name="t";
	    n_name.append(n_num);
	    t_name.append(n_num);
	    int_to_string(m,m_num,10);
	    n_name.append(m_num);
	    t_name.append(m_num);

	    //	    out << "     int " << t_name << "=" << m+1 << ";\n";
	    //	    out << "     int " << n_name << "=" << num << ";\n";
	    //                      out << "     int " << collect[b][n][m].name << "=floor(1/(abs(mcL[l]-" <<
	    //			n_name << "+" << t_name << "-1)+0.9));\n";
	    out << "     double " << collect[b][n][m].name << "=0.5*mcL[l]+0.5;\n";
	  }
	  else{
	    int num;
	    num=collect[b][n][m].num_specie/2;
	    out << "     l=index(i";
	    if(collect[b][n][m].shift[0] == 0) out << ",j";
	    if(collect[b][n][m].shift[0] < 0) out << collect[b][n][m].shift[0] << ",j";
	    if(collect[b][n][m].shift[0] > 0) out << "+" << collect[b][n][m].shift[0] << ",j";

	    if(collect[b][n][m].shift[1] == 0) out << ",k";
	    if(collect[b][n][m].shift[1] < 0) out << collect[b][n][m].shift[1] << ",k";
	    if(collect[b][n][m].shift[1] > 0) out << "+" << collect[b][n][m].shift[1] << ",k";

	    if(collect[b][n][m].shift[2] == 0) out << ",";
	    if(collect[b][n][m].shift[2] < 0) out << collect[b][n][m].shift[2] << ",";
	    if(collect[b][n][m].shift[2] > 0) out << "+" << collect[b][n][m].shift[2] << ",";

	    out << collect[b][n][m].shift[3] << ");\n";

	    string m_num,n_num;
	    string n_name,t_name;
	    int_to_string(n,n_num,10);
	    n_name="num";
	    t_name="t";
	    n_name.append(n_num);
	    t_name.append(n_num);
	    int_to_string(m,m_num,10);
	    n_name.append(m_num);
	    t_name.append(m_num);

	    //	    out << "     int " << t_name << "=" << m+1 << ";\n";
	    //	    out << "     int " << n_name << "=" << num << ";\n";
	    //	    if(m+1<=num) out << "     int " << collect[b][n][m].name << "=floor(1/(abs(mcL[l]-" <<
	    //			   n_name << "+" << t_name << "-1)+0.9));\n";
	    //	    else out << "     int " << collect[b][n][m].name << "=floor(1/(abs(mcL[l]-" <<
	    //		   n_name << "+" << t_name << ")+.9));\n";
	    out << "     double " << collect[b][n][m].name << "=0.5*mcL[l]+0.5;\n";
	  }
	}

	else{
	  //for spin basis
	  out << "     l=index(i";
	  if(collect[b][n][m].shift[0] == 0) out << ",j";
	  if(collect[b][n][m].shift[0] < 0) out << collect[b][n][m].shift[0] << ",j";
	  if(collect[b][n][m].shift[0] > 0) out << "+" << collect[b][n][m].shift[0] << ",j";

	  if(collect[b][n][m].shift[1] == 0) out << ",k";
	  if(collect[b][n][m].shift[1] < 0) out << collect[b][n][m].shift[1] << ",k";
	  if(collect[b][n][m].shift[1] > 0) out << "+" << collect[b][n][m].shift[1] << ",k";

	  if(collect[b][n][m].shift[2] == 0) out << ",";
	  if(collect[b][n][m].shift[2] < 0) out << collect[b][n][m].shift[2] << ",";
	  if(collect[b][n][m].shift[2] > 0) out << "+" << collect[b][n][m].shift[2] << ",";

	  out << collect[b][n][m].shift[3] << ");\n";
	  out << "     int " << collect[b][n][m].name << "=mcL[l]";
	  if(m==0) out << ";\n";
	  if(m>0){
	    for(int mm=0; mm<m; mm++){
	      out << "*mcL[l]";
	      if(mm==m-1) out << ";\n";
	    }
	  }
	}
      }
    }

    out << "\n";

    out << "     energy = energy";

    int nm=b;
    for(int np=1; np<montiplet[nm].orb.size(); np++){
      for(int no=0; no<montiplet[nm].orb[np].size(); no++){
	if(fabs(montiplet[nm].orb[np][no].eci) >= 0.000001){
	  double npoints=montiplet[nm].orb[np][no].equiv[0].point.size();
	  if(montiplet[nm].orb[np][no].eci < 0.0) out << montiplet[nm].orb[np][no].eci/npoints << "*(";
	  if(montiplet[nm].orb[np][no].eci > 0.0) out << "+" << montiplet[nm].orb[np][no].eci/npoints << "*(";
	  for(int ne=0; ne<montiplet[nm].orb[np][no].equiv.size(); ne++){
	    if(ne > 0) out << "+";
	    for(int n=0; n<montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){
	      mc_index tpoint;
	      for(int i=0; i<4; i++){
		tpoint.shift[i]=montiplet[nm].orb[np][no].equiv[ne].point[n].shift[i];
	      }
	      for(int i=0; i<collect[b].size(); i++){
		if(compare(tpoint,collect[b][i][0])){
		  int j;
		  j=montiplet[nm].orb[np][no].equiv[ne].point[n].bit;
		  //cout << "b: " << b << "\ti: " << i << "\tj: " << j << "\tname: " << collect[b][i][j].name << "\n";
		  out << collect[b][i][j].name;
		  if(n == montiplet[nm].orb[np][no].equiv[ne].point.size()-1){
		    out << "";
		  }
		  else out << "*";
		  break;
		}
	      }
	    }
	  }
	  out << ")";
	}
      }
    }
    out << ";\n";
    out << "     return energy;\n";
    out << "  }\n";
    out << "\n";
    out << "\n";
  }

  out << "}\n";

}


//************************************************************

// Routine adapted by John from Monte_Carlo::write_point_energy

void Monte_Carlo::write_point_corr(ofstream &out){


  // for each basis site, get all points that are accessed (points in all clusters that contain the basis site, and also have non-zero eci)

  out << "\n \n//************************************************************ \n \n";

  vector< vector< mc_index > > collect;

  for(int nm=0; nm<montiplet.size(); nm++){
    vector<mc_index> tcollect;
    for(int np=1; np<montiplet[nm].orb.size(); np++){
      for(int no=0; no<montiplet[nm].orb[np].size(); no++){
	if(abs(montiplet[nm].orb[np][no].eci) >= 0.000001){
	  for(int ne=0; ne<montiplet[nm].orb[np][no].equiv.size(); ne++){
            for(int n=0; n<montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){
	      mc_index tpoint;
	      for(int i=0; i<4; i++) tpoint.shift[i]=montiplet[nm].orb[np][no].equiv[ne].point[n].shift[i];

	      // check whether tpoint is already among the list in tcollect

	      bool already_present=false;
	      for(int i=0; i<tcollect.size(); i++){
		if(compare(tpoint,tcollect[i])){
		  already_present=true;
		  break;
		}
	      }
	      if(already_present == false){
		tpoint.l=tcollect.size();
		string p_num;
		int_to_string(tpoint.l,p_num,10);
		tpoint.name="p";
		tpoint.name.append(p_num);
		tcollect.push_back(tpoint);
	      }
	    }
          }
	}
      }
    }
    collect.push_back(tcollect);
  }


  // Begin writing Mote_Carlo::pointcorr

  out << "void Monte_Carlo::pointcorr(int i, int j, int k, int b){\n";


  out << "  int l; " << "\n";

  //Write the variable assignments -> spins of relevant sites are stored in doubles with names of form p0, p1, etc.

  for(int b=0; b< collect.size(); b++){
    out << "  if(b == " << b << "){\n";
    for(int n=0; n<collect[b].size(); n++){
      out << "     l=index(i";
      if(collect[b][n].shift[0] == 0) out << ",j";
      if(collect[b][n].shift[0] < 0) out << collect[b][n].shift[0] << ",j";
      if(collect[b][n].shift[0] > 0) out << "+" << collect[b][n].shift[0] << ",j";

      if(collect[b][n].shift[1] == 0) out << ",k";
      if(collect[b][n].shift[1] < 0) out << collect[b][n].shift[1] << ",k";
      if(collect[b][n].shift[1] > 0) out << "+" << collect[b][n].shift[1] << ",k";

      if(collect[b][n].shift[2] == 0) out << ",";
      if(collect[b][n].shift[2] < 0) out << collect[b][n].shift[2] << ",";
      if(collect[b][n].shift[2] > 0) out << "+" << collect[b][n].shift[2] << ",";

      out << collect[b][n].shift[3] << "); \n";
      out << "     double " << collect[b][n].name << "=mcL[l]; \n";
    }


    //write out the correlation formulas in terms of the pxx

    out << "\n";


    int nm=b;

    //First add the empty cluster
    out << "     AVcorr[0]+=1.0/" << collect.size() << ";\n";

    for(int np=1; np<montiplet[nm].orb.size(); np++){
      for(int no=0; no<montiplet[nm].orb[np].size(); no++){
        if(fabs(montiplet[nm].orb[np][no].eci) >= 0.000001){

	  int mult_num=0;
	  bool break_flag=false;
	  for(int i=0; i<basiplet.orb.size(); i++){
	    for(int j=0; j<basiplet.orb[i].size(); j++){
	      if(montiplet[nm].index[np][no]==mult_num){
		mult_num=basiplet.orb[i][j].equiv.size();
		break_flag=true;
	      }
	      if(break_flag) break;
	      mult_num++;
	    }
	    if(break_flag) break;
	  }


	  out << "     AVcorr[" << montiplet[nm].index[np][no] << "]+=(";

	  for(int ne=0; ne<montiplet[nm].orb[np][no].equiv.size(); ne++){
	    if(ne > 0) out << "+";
	    for(int n=0; n<montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){

	      //find the name of the point
	      mc_index tpoint;
	      for(int i=0; i<4; i++) tpoint.shift[i]=montiplet[nm].orb[np][no].equiv[ne].point[n].shift[i];
	      for(int i=0; i<collect[b].size(); i++){
		if(compare(tpoint,collect[b][i])){
		  for(int j=0; j<montiplet[nm].orb[np][no].equiv[ne].point[n].bit+1; j++){
		    out << collect[b][i].name;
		    if(n == montiplet[nm].orb[np][no].equiv[ne].point.size()-1 && j == montiplet[nm].orb[np][no].equiv[ne].point[n].bit) out << "";
		    else out << "*";
		  }
		  break;
		}
	      }

	    }
	  }
	  out << ")";

	  // Divide by multiplicity and/or size of each cluster
	  if(mult_num>1)
	    out << "/" << mult_num*np;
	  else if(np>1)
	    out << "/" << np;
	  out << ";\n";

	}
      }
    }
    out << "     return;\n";
    out << "  }\n";
    out << "\n";
    out << "\n";
  }

  out << "}\n";

}


//************************************************************

void Monte_Carlo::write_monte_h(string class_file){
  ifstream in;
  in.open(class_file.c_str());

  if(!in){
    cout << "cannot open the " << class_file << " file \n";
    cout << "no monte.h created \n";
    return;
  }

  ofstream out;
  out.open("monte.h");

  if(!out){
    cout << "unable to create/open monte.h\n";
    cout << "no monte.h created \n";
    return;
  }

  bool point_energy_written=false;
  string line;
  while(getline(in,line) && !point_energy_written){
    string check=line.substr(0,31);
    if(check == "double Monte_Carlo::pointenergy"){
      in.close();
      write_point_energy(out);
      out << "\n";
      out << "\n";
      write_normalized_point_energy(out);
      out << "\n";
      out << "\n";
      write_point_corr(out);
      //out << "\n";   // added by jishnu
      //out << "\n";   // added by jishnu
      //write_environment_bool_table(out);   // added by jishnu
      //out << "\n";   // added by jishnu
      //out << "\n";   // added by jishnu
      //write_evaluate_bool(out);   // added by jishnu
      point_energy_written=true;
      out.close();
    }
    else{
      out << line;
      out << "\n";
    }
  }

  if(!point_energy_written){
    in.close();
    write_point_energy(out);
    out << "\n";
    out << "\n";
    write_normalized_point_energy(out);
    out << "\n";
    out << "\n";
    write_point_corr(out);
    //out << "\n";   // added by jishnu
    //out << "\n";   // added by jishnu
    //write_environment_bool_table(out);   // added by jishnu
    //out << "\n";   // added by jishnu
    //out << "\n";   // added by jishnu
    //write_evaluate_bool(out);   // added by jishnu
    point_energy_written=true;
    out.close();
  }

  in.close();
  return;


}


//************************************************************
//************************************************************

void Monte_Carlo::write_monte_xyz(ostream &stream){
  //first update the monte carlo structure to have the correct occupancies
  //then print the monte carlo structure
  for(int i=0; i<Monte_Carlo_cell.atom.size(); i++){
    int l=Monte_Carlo_cell.atom[i].bit;
    for(int j=0; j<Monte_Carlo_cell.atom[i].compon.size(); j++){
      if(mcL[l] == Monte_Carlo_cell.atom[i].compon[j].spin) Monte_Carlo_cell.atom[i].occ=Monte_Carlo_cell.atom[i].compon[j];
    }
  }

  Monte_Carlo_cell.write_struc_xyz(stream, conc);

}




//************************************************************

bool Monte_Carlo::compatible(structure struc, int &ndi, int &ndj, int &ndk){

  // checks whether the given MC-dimensions (di,dj,dk) are compatible with the struc unit cell
  // if not, new ones (ndi,ndj,ndk which are the smallest ones just larger than di,dj,dk) are
  // suggested that are compatible


  double mclat[3][3],struclat[3][3],inv_struclat[3][3],strucTOmc[3][3];

  //since struc could correspond to a relaxed structure, its lat[][] may not be integer
  //multiples of the prim.lat[][]
  //therefore, we first find the closest integer multiples of prim.lat[][]
  //and then calculate the struclat[][] as these integer multiples of prim.lat[][]

  struc.generate_slat(prim);   // finds the closest integer multiples of struc.lat in terms of prim.lat and puts it into struc.slat

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      struclat[i][j]=0.0;
      for(int k=0; k<3; k++){
        struclat[i][j]=struclat[i][j]+struc.slat[i][k]*prim.lat[k][j];
      }
    }
  }

  inverse(struclat,inv_struclat);

  ndi=di; ndj=dj; ndk=dk;

  int int_rows;
  do{

    for(int j=0; j<3; j++){
      mclat[0][j]=ndi*prim.lat[0][j];
      mclat[1][j]=ndj*prim.lat[1][j];
      mclat[2][j]=ndk*prim.lat[2][j];
    }

    //Determine the matrix that relates the Monte Carlo cell (mclat) to the struc cell (struclat)

    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
        strucTOmc[i][j]=0.0;
	for(int k=0; k<3; k++){
          strucTOmc[i][j]=strucTOmc[i][j]+mclat[i][k]*inv_struclat[k][j];
        }
      }
    }

    //Test whether the rows of strucTOmc[][] are all integers (within numerical noise)
    //if not, increment the mc dimensions corresponding to the non-integer rows

    int_rows=0;
    if(!is_integer(strucTOmc[0])) ndi++;
    else int_rows++;
    if(!is_integer(strucTOmc[1])) ndj++;
    else int_rows++;
    if(!is_integer(strucTOmc[2])) ndk++;
    else int_rows++;

  }while(int_rows !=3);

  if(ndi == di && ndj == dj && ndk == dk) return true;
  else return false;

}

//************************************************************

void Monte_Carlo::initialize(structure init_struc){

  init_struc.generate_slat(prim);
  init_struc.map_on_expanded_prim_basis(prim);


  //create the Monte_Carlo_cell structure

  for(int i=0; i<200; i++) Monte_Carlo_cell.title[i]=prim.title[i];
  Monte_Carlo_cell.scale=prim.scale;
  for(int j=0; j<3; j++){
    Monte_Carlo_cell.lat[0][j]=di*prim.lat[0][j];
    Monte_Carlo_cell.lat[1][j]=dj*prim.lat[1][j];
    Monte_Carlo_cell.lat[2][j]=dk*prim.lat[2][j];
  }

  //determine the relation between the Monte_Carlo_cell.lat[][] and the init_struc.lat[][]
  //this goes into Monte_Carlo_cell.slat[][]
  //then expand the init_struc and fill up the Monte_Carlo_cell with it
  //It is assumed that the Monte_Carlo_cell and init_struc are compatible with each other

  Monte_Carlo_cell.generate_slat(init_struc);
  Monte_Carlo_cell.expand_prim_basis(init_struc);


  //for each site within the Monte_Carlo_cell (with more than two components) determine the shift indices (indicates indices of unit cell and basis)
  //to do that we want the Monte_Carlo_cell coordinates in the primitive cell coordinate system

  Monte_Carlo_cell.generate_slat(prim);
  Monte_Carlo_cell.get_trans_mat();

  ////////////////////////////////////////////////////////////////////////////////
  //swoboda
  //for(int na=0; na<2000; na++){ //Monte_Carlo_cell.atom.size(); na++){
  //    cout << "atom[" << na << "]: " << Monte_Carlo_cell.atom[na].bit << "\t";
  //}
  //cout << "\nsize of Monte Carlo cell: " << Monte_Carlo_cell.atom.size() << "\n";
  ////////////////////////////////////////////////////////////////////////////////

  for(int na=0; na<Monte_Carlo_cell.atom.size(); na++){
    if(Monte_Carlo_cell.atom[na].compon.size() >= 2){
      atompos hatom=Monte_Carlo_cell.atom[na];
      conv_AtoB(Monte_Carlo_cell.StoP,Monte_Carlo_cell.atom[na].fcoord,hatom.fcoord);
      get_shift(hatom,basis);

      //assign the atom[na].bit variable the index within the mcL array for this site

      int i=hatom.shift[0];
      int j=hatom.shift[1];
      int k=hatom.shift[2];
      int b=hatom.shift[3];

      int l=index(i,j,k,b);
      Monte_Carlo_cell.atom[na].bit=l;

      mcL[l]=Monte_Carlo_cell.atom[na].occ.spin;
    }
  }

  // xph: construct nearest neighbor list (all the possible movements are stored here)
  // todo: calculate automatically from PRIM
  //double moves[8][3];     //all the possible movements
  // add 6 octahedral to octahedral inlayer hops for Li
  // moves infractional coordinates for close-packed cubic prim cell (LiNiO2) 
  double moves[nmoves][3] = 
  { { 0.25,  0.25,  0.25},
    {-0.25, -0.25, -0.25},
    { 0.75, -0.25, -0.25},
    {-0.75,  0.25,  0.25},
    { 0.25,  0.25, -0.75},
    {-0.25, -0.25,  0.75},
    {-0.25,  0.75, -0.25},
    { 0.25, -0.75,  0.25},
  //o2o move in Li layer. xph warning: POSCAR has to define Li layer perpendicular to the b axis.
    { 1.0, 0.0,  0.0},
    { 0.0, 0.0,  1.0},
    {-1.0, 0.0,  1.0},
    {-1.0, 0.0,  0.0},
    { 0.0, 0.0, -1.0},
    { 1.0, 0.0, -1.0},
  };

  double endpoint[3];  // temporary viable for basis[b]+moves[movei]
  int trans[3]; // trans stores which neighboring cell the move leads to.
	        // if trans is {0,0,1}, this move connects basis[b] 
	        // and basis[bt] in the next periodi cell along c-axis.
  movemap.resize(db); // movemap is a public variable: 
                      // movemap[b][movei][0] is the connected basis for basis[b]
                      // movemap[b][movei][1:4] is the trans 
  for(int b=0; b<db; b++){
    movemap[b].resize(nmoves);
    for(int movei=0; movei<nmoves; movei++){
      movemap[b][movei].resize(4, -1);
      for(int ti=0; ti<3; ti++){
        endpoint[ti] = basis[b].fcoord[ti] + moves[movei][ti];
      }
      for(int bt=0; bt<db; bt++){
        // no o2o move for tetrahedral sites
        // xph warning: PRIM has to define bt>0  as tetrahedral sites.
        if(bt>0 and movei>7) continue;
        if(compare(endpoint, basis[bt].fcoord, trans)){
          movemap[b][movei][0] = bt;
          movemap[b][movei][1] = trans[0];
          movemap[b][movei][2] = trans[1];
          movemap[b][movei][3] = trans[2];
        }
      }
    }
  }

  // get spins of different species for futher reference
  Lispin  = basis[0].get_spin("Li");
  Nispin  = basis[0].get_spin("Ni");
  Vacspin = basis[0].get_spin("Vac");

  // initialize update lists before the loop
  get_interaction_list();
  get_partial_update();

  // initialize hash table for looking up calculated energies. 
  energy_map.resize(db);
  //ht_size = 29; // the hash table size is 2^ht_size.
  //htcap = 1<<ht_size; // 2**ht_size
  ht_size.resize(db);
  htcap.resize(db);
  htload.resize(db, 0);
  htncol.resize(db, 0);
  //pair<bitset<224>, double> ptmp (0, 1000.0);
  //array<uint64_t, 4> ilenv;
  array<uint64_t, 5> ilenv;
  ilenv.fill(0);
  //pair<array<uint64_t, 4>, double> ptmp (ilenv, 1000.0);
  pair<array<uint64_t, 5>, double> ptmp (ilenv, 1000.0);
  for(int b=0; b<db; b++){
    if(b==0){
       ht_size[b] = 30;
    } else {
       ht_size[b] = 29;
    }
    htcap[b] = 1<<ht_size[b]; // 2**ht_size
    energy_map[b].resize(htcap[b], ptmp);
  }
  //cout << "interaction size" << interaction_list[0].size();

  // xph: initilize the rateTable
  // rateTable for all possible moves of all sites
  rateTable.resize(nmcL);
  for(int l=0; l<nmcL; l++){
    // grand everywhere
    //rateTable[l].resize(nmoves+1, 0.0);
    // canonical only
    //rateTable[l].resize(nmoves, 0.0);
    
    rateTable[l].resize(nmoves, 0.0);
    // surface sites have two more moves: Li comes in and Li goes out
    // todo: read surface index and basis from input; generalize to any surface.
    int ii=ltoi[l];
    // three layers can do grand moves
    int right_surface = 6;
    if(ii > 0 and ii < right_surface){
      // xph v1: no Ni grand move for this version
      rateTable[l].resize(nmoves+1, 0.0);
    } else if(ii==0 or ii==right_surface){
      // Ni grand move surface is behind the Li grand surface (0, 1, 2)
      rateTable[l].resize(nmoves+2, 0.0);
    }
  }

}



//************************************************************

void Monte_Carlo::initialize(concentration in_conc){

  //visit each site of the cell
  //randomly assign a spin with the probability in conc to the Monte Carlo cell

  for(int i=0; i < di; i++){
    for(int j=0; j < dj; j++){
      for(int k=0; k < dk; k++){
	for(int b=0; b<db; b++){
	  //check whether the basis site is regular (as opposed to activated)
	  //SPECIFIC FOR LITIS2
	  if(basis[b].bit == 0){
	    //if(basis[b].basis_flag == '0'){
	    int l=index(i,j,k,b);
	    //get the basis to concentration link to determine which spins can occupy that site
	    int c=basis_to_conc[b];

	    double p=ran0(idum);
	    double sum=0.0;
	    for(int d=0; d<in_conc.compon[c].size(); d++){
	      sum=sum+in_conc.occup[c][d];
	      if(p <= sum){
		mcL[l]=in_conc.compon[c][d].spin;
		break;
	      }
	    }
	  }
	}
      }
    }
  }
  //calculate the new concentration
  calc_concentration();
  calc_sublat_concentration();
  calc_num_atoms();
}

//************************************************************

//************************************************************
//Added by Aziz : Beginning
//************************************************************

void Monte_Carlo::initialize_1_specie(double in_conc){

  //This routine fills the bulk to the specified concentration in_conc (double format)
  //This routine assumes that there is only 1 type of specie in the system
  //DO NOT USE FOR MULTI-SPECIES SYSTEMS

  //visit each site of the cell
  //randomly assign a spin with the probability in conc to the Monte Carlo cell


  for(int i=0; i < di; i++){
    for(int j=0; j < dj; j++){
      for(int k=0; k < dk; k++){
	    for(int b=0; b<db; b++){
	      //check whether the basis site is regular (as opposed to activated)
	      //SPECIFIC FOR LITIS2
	      if(basis[b].bit == 0){
	      //if(basis[b].basis_flag == '0'){
	        int l=index(i,j,k,b);


	        double p=ran0(idum);

            if(p <= in_conc){
		      mcL[l]=1;
	        }
		    else { //Empty site that is not in vacuum
		      mcL[l]=-1;
	        }
	      }
        }
      }
    }
  }
  //calculate the new concentration
  calc_concentration();
  calc_sublat_concentration(); //Modify in order to consider only bulk sites
  calc_num_atoms();
}



//************************************************************
//Added by Aziz : End
//************************************************************

//************************************************************

void Monte_Carlo::initialize_1vac(concentration in_conc){
  ////////////////////////////////////////////////////////////////////////////////
  //added by Ben Swoboda
  //initialize changed so that only 1 site in structure becomes a vacancy.

  int count=0;

  for(int i=0; i<di; i++){
    for(int j=0; j<dj; j++){
      for(int k=0; k<dk; k++){
	for(int b=0; b<db; b++){
	  //check if basis site is a lattice site
	  if(basis[b].bit==0){
	    int l=index(i,j,k,b);
	    int c=basis_to_conc[b];
	    if(count==0){
	      mcL[l]=in_conc.compon[0][1].spin;
	      count++;
	    }
	    else {
	      mcL[l]=in_conc.compon[0][0].spin;
	    }
	  }
	}
      }
    }
  }

  //calculate the new concentration
  calc_concentration();
  calc_sublat_concentration();
  calc_num_atoms();

}

////////////////////////////////////////////////////////////////////////////////
//************************************************************

void Monte_Carlo::calc_energy(double &energy){
  energy=0.0;
  for(int i=0; i<di; i++){
    for(int j=0; j<dj; j++){
      for(int k=0; k<dk; k++){
	energy=energy+eci_empty;
        for(int b=0; b<db; b++){
          double temp=normalized_pointenergy(i,j,k,b);
          energy=energy+temp;
        }
      }
    }
  }
}



//************************************************************

void Monte_Carlo::calc_num_atoms(){
  //determines how many of each component there are in the monte carlo cell

  for(int i=0; i<num_atoms.compon.size(); i++){
    for(int j=0; j<num_atoms.compon[i].size(); j++){
      num_atoms.occup[i][j]=0.0;
    }
  }
  //go through the Monte Carlo lattice
  for(int i=0; i<di; i++){
    for(int j=0; j<dj; j++){
      for(int k=0; k<dk; k++){
        for(int b=0; b<db; b++){
          //determine the concentration unit
          int c=basis_to_conc[b];
          int l=index(i,j,k,b);
          for(int m=0; m<num_atoms.compon[c].size(); m++){
            if(mcL[l] == num_atoms.compon[c][m].spin) num_atoms.occup[c][m]++;
          }
        }
      }
    }
  }

}

//************************************************************

void Monte_Carlo::calc_concentration(){

  for(int i=0; i<conc.compon.size(); i++){
    for(int j=0; j<conc.compon[i].size(); j++){
      conc.occup[i][j]=0.0;
    }
  }
  //go through the Monte Carlo lattice
  for(int i=0; i<di; i++){
    for(int j=0; j<dj; j++){
      for(int k=0; k<dk; k++){
        for(int b=0; b<db; b++){
	  //only calculate the concentration over the regular sites
	  //SPECIFIC FOR LITIS2
	  if(basis[b].bit == 0){
	    // if(basis[b].basis_flag == '0'){
	    //determine the concentration unit
	    int c=basis_to_conc[b];
	    int l=index(i,j,k,b);
	    for(int m=0; m<conc.compon[c].size(); m++){
	      if(mcL[l] == conc.compon[c][m].spin) conc.occup[c][m]++;
	    }
	  }
        }
      }
    }
  }


  for(int i=0; i<conc.compon.size(); i++){
    //number of sublattices with the i'th concentration unit
    int n=0;
    for(int ii=0; ii<conc_to_basis[i].size(); ii++)
      //SPECIFIC FOR LITIS2
      // if(basis[conc_to_basis[i][ii]].basis_flag == '0') n++;
      if(basis[conc_to_basis[i][ii]].bit == 0) n++;

    //number of sites in the crystal with the i'th concentration unit
    n=n*di*dj*dk;
    if(n != 0){
      for(int j=0; j<conc.compon[i].size(); j++){
	conc.occup[i][j]=conc.occup[i][j]/n;
      }
    }
  }
}


//************************************************************

void Monte_Carlo::calc_sublat_concentration(){

  for(int i=0; i<sublat_conc.compon.size(); i++){
    for(int j=0; j<sublat_conc.compon[i].size(); j++){
      sublat_conc.occup[i][j]=0.0;
    }
  }
  //go through the Monte Carlo lattice
  for(int i=0; i<di; i++){
    for(int j=0; j<dj; j++){
      for(int k=0; k<dk; k++){
        for(int b=0; b<db; b++){
	  //only calculate the concentration over the regular sites
	  if(basis[b].bit == 0){
	    //determine the concentration unit
	    int c=basis_to_sublat[b];
	    int l=index(i,j,k,b);
	    for(int m=0; m<sublat_conc.compon[c].size(); m++){
	      if(mcL[l] == sublat_conc.compon[c][m].spin) sublat_conc.occup[c][m]++;
	    }
	  }
        }
      }
    }
  }


  for(int i=0; i<sublat_conc.compon.size(); i++){
    //number of sublattices with the i'th concentration unit
    int n=0;
    for(int ii=0; ii<sublat_to_basis[i].size(); ii++)
      if(basis[sublat_to_basis[i][ii]].bit == 0)n++;

    //number of sites in the crystal with the i'th concentration unit
    n=n*di*dj*dk;
    if(n != 0){
      for(int j=0; j<sublat_conc.compon[i].size(); j++){
	sublat_conc.occup[i][j]=sublat_conc.occup[i][j]/n;
      }
    }
  }
}





//************************************************************

void Monte_Carlo::update_num_hops(int l, int ll, int b, int bb){
  int c=basis_to_conc[b];
  int cc=basis_to_conc[bb];

  for(int i=0; i<num_hops.compon[c].size(); i++){
    if(num_hops.compon[c][i].spin == mcL[l]) num_hops.occup[c][i]++;
  }

  for(int i=0; i<num_hops.compon[cc].size(); i++){
    if(num_hops.compon[cc][i].spin == mcL[ll]) num_hops.occup[cc][i]++;
  }
  return;

}




//************************************************************

double Monte_Carlo::calc_grand_canonical_energy(chempot mu){
  double energy;
  calc_energy(energy);
  calc_num_atoms();

  for(int i=0; i<num_atoms.compon.size(); i++){
    for(int j=0; j<num_atoms.compon[i].size(); j++){
      energy=energy-mu.m[i][j]*num_atoms.occup[i][j];
    }
  }
  return energy;

}

//************************************************************
// xph: routine to get sites that interact with the center site,
//      adapted from Monte_Carlo::write_point_corr.
//      will be called to get sites that need to be updated after a hop.
void  Monte_Carlo::get_interaction_list(){
  // for each basis site, get all points that are accessed (points in all clusters that contain the basis site, and also have non-zero eci)

  for(int nm=0; nm<montiplet.size(); nm++){
    vector<mc_index> tcollect;
    for(int np=1; np<montiplet[nm].orb.size(); np++){
      for(int no=0; no<montiplet[nm].orb[np].size(); no++){
	if(abs(montiplet[nm].orb[np][no].eci) >= 0.000001){
	  for(int ne=0; ne<montiplet[nm].orb[np][no].equiv.size(); ne++){
            for(int n=0; n<montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){
	      mc_index tpoint;
	      for(int i=0; i<4; i++) tpoint.shift[i]=montiplet[nm].orb[np][no].equiv[ne].point[n].shift[i];

	      // check whether tpoint is already among the list in tcollect

	      bool already_present=false;
	      for(int i=0; i<tcollect.size(); i++){
		if(compare(tpoint,tcollect[i])){
		  already_present=true;
		  break;
		}
	      }
	      if(already_present == false){
		tpoint.l=tcollect.size();
		tpoint.name="binds";  // labelled as "binds" if the site is in the interaction zone
		tcollect.push_back(tpoint);
	      }
	    }
          }
	}
      }
    }
    interaction_list.push_back(tcollect);
  }
}
//************************************************************
// xph: routine to get sites that connect to the interaction zone by one move,
void  Monte_Carlo::get_partial_update(){
  // xph: sites and their specific moves that starts outsite but end into the interaction sphere (6A, 2 unit cells)
  // loop over the interaction_list for all basis 
  for(int b=0; b<interaction_list.size(); b++){
  // creat a hash table for the interaction_list for basis "b"
  vector<int> hash_interaction;
  hash_interaction.resize(nmcL, 0);
  for(int i=0; i<interaction_list[b].size(); i++){
    int shi = interaction_list[b][i].shift[0];
    int shj = interaction_list[b][i].shift[1];
    int shk = interaction_list[b][i].shift[2];
    int shb = interaction_list[b][i].shift[3];
    int shl = index(shi, shj, shk, shb);
    hash_interaction[shl] = 1;
    //cout << i << ":" << shl << endl;
  }

  // xph warning: update region, [-3, 4) here, need to be determined from the interaction radius and hop length.
  vector<mc_index> tcollect;
  for(int ui=-3; ui<4; ui++){
    for(int uj=-3; uj<4; uj++){
      for(int uk=-3; uk<4; uk++){
        for(int ub=0; ub<db; ub++){
          // skip if the initial point is already in the interaction list
          int ul = index(ui, uj, uk, ub);
          if(hash_interaction[ul] == 1) continue;
 
          int site_present=0;
	  mc_index tupp;
          for(int um=0; um<nmoves; um++){
            // check if the end point is in the interaction list
            int epb = movemap[ub][um][0]; // basis index of the destination
            if(epb == -1) continue;  // invalid move
            int epi = ui + movemap[ub][um][1]; // cell index of the destination
            int epj = uj + movemap[ub][um][2];
            int epk = uk + movemap[ub][um][3];
            int epl = index(epi, epj, epk, epb);
            if(hash_interaction[epl] == 1){
                  if(site_present==0){
                    // store the site and movement that needs an update in tupp
                    tupp.shift[0] = ui;
                    tupp.shift[1] = uj;
                    tupp.shift[2] = uk;
                    tupp.shift[3] = ub;
                    string movetags="00000000000000";
                    const string tag="1";
                    movetags.replace(um,1,tag);
                    tupp.name = movetags;
                  } else {
                    // only update the movement tag
                    const string tag="1";
                    tupp.name.replace(um,1,tag);
                  }
		  site_present++;
	    }
          }
          if(site_present > 0){
            //cout << tupp.shift[0] << tupp.shift[1] << tupp.shift[2] << tupp.shift[3] << endl;
            //cout << tupp.name << endl;
            //cout << "****************************"<< endl;
	    tcollect.push_back(tupp);
          }
        }
      }
    }
  }
  partial_update_list.push_back(tcollect);
  }
}

//************************************************************
//xph: routine to update rateTable[l]
void Monte_Carlo::update_rate_table(int l, double beta, bitset<nmoves> movetags){
    int i=ltoi[l];
    int j=ltoj[l];
    int k=ltok[l];
    int b=ltob[l];
    int tspin   = mcL[l]; // save the current spin temporarily
    //int Lispin  = basis[b].get_spin("Li");
    //int Nispin  = basis[b].get_spin("Ni");
    //int Vacspin = basis[b].get_spin("Vac");
    double Ea0ini = (mcL[l] == Nispin) ? Ea0_Ni: Ea0_Li;
    double Ea0 = Ea0ini;
    double delta_energy_first; // delta_energy when the first site flips to vacancy

    if(mcL[l] != Vacspin){
       // get point energy before the spin flip for the first site, Li/Ni->Vac
       //double en_before=pointenergy(i,j,k,b);
       double en_before=get_pointenergy(i,j,k,b);
       // get point energy after the spin flip for the first site
       mcL[l]=Vacspin;
       //double en_after=pointenergy(i,j,k,b);
       double en_after=get_pointenergy(i,j,k,b);
       delta_energy_first = en_after-en_before;
       mcL[l]=tspin;
    }

    // dE is only calculated for the move towards a vacancy.
    for(int mj=0; mj<nmoves; mj++){
       // for vacancy site keep the rate as 0.0, unless on surface(mj>=nmoves)
       if(mcL[l] == Vacspin){
         rateTable[l][mj] = 0.0;
         continue; 
       }

       // no need to update
       if(not movetags.test(mj)) continue;

       int b_2 = movemap[b][mj][0]; // basis of the destination site
       if(b_2==-1) continue;        // no site exists along this move
       int i_2 = i + movemap[b][mj][1]; // cell index of the destination
       int j_2 = j + movemap[b][mj][2];
       int k_2 = k + movemap[b][mj][3];
       int l_2=index(i_2,j_2,k_2,b_2);
       //int Nispin_2 = basis[b_2].get_spin("Ni");
       //int Vacspin_2 = basis[b_2].get_spin("Vac");

       // v1.6: direct octahedral to octahedral hops are only for Li
       // stored as move[8:13]
       //if((mcL[l] == Nispin or mcL[l_2] != Vacspin_2) and mj > 7 and mj < 14){
       if((mcL[l] == Nispin or mcL[l_2] != Vacspin) and mj > 7 and mj < 14){
         rateTable[l][mj] = 0.0;
         continue; 
       } else if(mcL[l] == Lispin and mj > 7 and mj < 14){
         Ea0 = Ea0_Li_o2o;
         int gateNi = 0; // number of Ni at the gate positions. Barrier increases only when the two gate sites are both Ni.
         int gateVac = 0; // number of Vac at the gate positions. 
         for(int dm=-1; dm<2; dm+=2){
           int mlg = mj + dm; // o2o (octahedral to octahedral) moves are stored in a clockwise order
           // 8-13 are the six o2o moves
           if(mlg == 7){
             mlg = 13;
           } else if(mlg == 14){
             mlg = 8;
           }
           int b_lg = movemap[b][mlg][0]; // basis of the left gate site
                                          // or right gate site when dm==1
           int i_lg = i + movemap[b][mlg][1]; // cell index of the left gate
           int j_lg = j + movemap[b][mlg][2];
           int k_lg = k + movemap[b][mlg][3];
           int l_lg = index(i_lg,j_lg,k_lg,b_lg);
           //int Nispin_lg = basis[b_lg].get_spin("Ni");
           //int Vacspin_lg = basis[b_lg].get_spin("Vac");
           //if(mcL[l_lg] == Vacspin_lg){
           if(mcL[l_lg] == Vacspin){
             gateVac += 1;
             break; 
           //} else if(mcL[l_lg] == Nispin_lg){
           } else if(mcL[l_lg] == Nispin){
             gateNi += 1;  // Ni at gate. Two Ni increase the barrier, but one Ni does not.  
           } else {
             // check the neighbors of the gate sites
             // if not backed by other Li/Ni, easy to pass.
             for(int dmg=0; dmg<2; dmg++){
               int mlgg = mlg + dmg*dm;
               if(mlgg == 7){
                 mlgg = 13;
               } else if(mlgg == 14){
                 mlgg = 8;
               }
               int b_lgg = movemap[b_lg][mlgg][0]; // basis of the neighbor of the left gate site
                                                   // or the right gate site when dm==1
               int i_lgg = i_lg + movemap[b_lg][mlgg][1]; // cell index of the neighbor of the left gate
               int j_lgg = j_lg + movemap[b_lg][mlgg][2];
               int k_lgg = k_lg + movemap[b_lg][mlgg][3];
               int l_lgg = index(i_lgg,j_lgg,k_lgg,b_lgg);
               int Vacspin_lgg = basis[b_lgg].get_spin("Vac");
               if(mcL[l_lgg] == Vacspin_lgg){
                 Ea0 -= 0.06;
               }
             }
           }
         }
         if(gateNi == 2){
           Ea0 = 0.76; // Two Ni at the o2o gate.
         } else if (gateVac > 0){
           rateTable[l][mj] = 0.0; // o2o merges to  the tetrahedral path when one gate site is not occupied
           continue; 
         }
       }

       //if(mcL[l_2] == Vacspin_2){
       if(mcL[l_2] == Vacspin){
         // save the spin of the destination site
         int tspin_2 = mcL[l_2];
         // take en_before out of the move loop because it doesn't depend on the end point of the move
         // get point energy before the spin flip for the first site, Ni->Vac
         //double en_before=pointenergy(i,j,k,b);

         // xph: warning: all the sites have to use the same spin for the same element
         mcL[l]=tspin_2;
         // take en_after out of the move loop because the end point is always a vacancy 
         // possible extension: if exchange with a non-vacancy site is allowed, the en_after should be kept in the loop
         // get point energy after the spin flip for the first site
         //double en_after=pointenergy(i,j,k,b);
         //double delta_energy = en_after-en_before;
         double delta_energy = delta_energy_first; 

         // get point energy before the spin flip for the second site, Vac->Ni
         //double en_before=pointenergy(i_2,j_2,k_2,b_2);
         double en_before=get_pointenergy(i_2,j_2,k_2,b_2);

         // get point energy after the spin flip for the second site
         // xph: warning: all the sites have to use the same spin for the same element
         mcL[l_2]=tspin;
         //double en_after=pointenergy(i_2,j_2,k_2,b_2);
         double en_after=get_pointenergy(i_2,j_2,k_2,b_2);
         delta_energy += en_after-en_before;
         //cout << "move:" << l << "->" << l_2 << "\n";
         //cout << "dE:" << delta_energy << "\n";
         
         // restore spins in mcL[l] and mcL[l_2]
         mcL[l]   = tspin;
         mcL[l_2] = tspin_2;

         // barrier and rate
         double Ea = max(0.0, Ea0 + 0.5*delta_energy);
         if(delta_energy>0) Ea = max(Ea, delta_energy);
         double rate = freq * exp(-Ea*beta);
         rateTable[l][mj] = rate;
         // debug Ea: remove later!
         //if(Ea<0.3) cout << l << " " << mj << ": " << Ea << " | " << endl;
       } else {
         rateTable[l][mj] = 0.0;
       }
    }

    // rates for grandcanonical move, sites on surface
    Ea0 = Ea0ini;
    for(int mk=nmoves; mk<rateTable[l].size(); mk++){
      // store Li (mk-nmoves==0) or Ni (mk-nmoves==1) grandcanonical move 
      // for mk-nmoves==0, only Li-Vac or Vac-Li swap is allowed.
      // for mk-nmoves==1, only Ni-Vac or Vac-Ni swap is allowed.
      // note:keep this part consistent with the move after selection
      int allowspin = (mk-nmoves == 0) ? Lispin: Nispin;
      if(mcL[l] != allowspin and mcL[l] != Vacspin){
        rateTable[l][mk] = 0.0;
        continue;
      }

      // Ni_grand: Ni can only come in, no Ni out.
      if(mcL[l] == Nispin){
        rateTable[l][mk] = 0.0;
        continue;
      }
      // Ni_grand: sites where Ni come in is not open to Li grand move.
      if(allowspin == Lispin and rateTable[l].size()-nmoves == 2){
        rateTable[l][mk] = 0.0;
        continue;
      }

      // no need to update
      if(movetags.count() < movetags.size()) continue; 

      // determine index of current occupant at site l
      int co=basis[b].iflip(mcL[l]);
      // determine index of next step occupant at site l
      int f=-1;
      for(int ti=0; ti<basis[b].flip[co].size(); ti++){
        int newspin=basis[b].flip[co][ti];
        if(newspin == allowspin or newspin == Vacspin){
          f=ti;
          break;
        }
      }

      //double en_before=pointenergy(i,j,k,b)-basis[b].mu[co];
      double en_before=get_pointenergy(i,j,k,b)-basis[b].mu[co];
      mcL[l]=basis[b].flip[co][f];
      int no=basis[b].iflip(mcL[l]);
      //double en_after=pointenergy(i,j,k,b)-basis[b].mu[no];
      double en_after=get_pointenergy(i,j,k,b)-basis[b].mu[no];
      double delta_energy = en_after-en_before;
      // restore the spin in mcL[l] 
      mcL[l]=tspin;

      // barrier and rate
      double Ea = max(0.0, Ea0 + 0.5*delta_energy);
      if(delta_energy>0) Ea = max(Ea, delta_energy);
      // assume the prefactor to be 1e13, setting freq=1e7 gives time in the unit of 1e-6s
      double rate = freq * exp(-Ea*beta);
      rateTable[l][mk] = rate;
      // debug Ea: remove later!
      //cout << l << " " << mk << ": " << Ea << " | " << endl;
    }
}

// xph: hash a configuration to a hashtable index
// http://create.stephan-brumme.com/fnv-hash/
// fnv1a, one byte
// default values recommended by http://isthe.com/chongo/tech/comp/fnv/
 const uint32_t Prime = 0x01000193; //   16777619
 const uint32_t Seed  = 0x811C9DC5; // 2166136261
/// hash a single byte
 inline uint32_t fnv1a(unsigned char oneByte, uint32_t hash = Seed)
 {
   return (oneByte ^ hash) * Prime;
 }
/// hash a block of memory
 uint32_t fnv1a(const void* data, size_t numBytes, uint32_t hash = Seed)
 {
   // assert(data);
   const unsigned char* ptr = (const unsigned char*)data;
   while (numBytes--)
     hash = fnv1a(*ptr++, hash);
   return hash;
 }

double Monte_Carlo::get_pointenergy(int i, int j, int k, int b){
  // xph: first check if the local environment has been calculated and stored in energy_map,
  //      if not call pointenergy() and add the result to energy_map.
  int l = index(i, j, k, b);
  if(mcL[l] == 0) return 0.0;

  // slowest
  /*
  bitset<224> blenv;
  vector<bitset<64>> bint_compare(4);
  blenv |= (mcL[l]+1) & 0x3;
  bint_compare[0] |= (mcL[l]+1) & 0x3;
  for(int ui=0; ui<interaction_list[b].size(); ui++){
    int ti = i + interaction_list[b][ui].shift[0];
    int tj = j + interaction_list[b][ui].shift[1];
    int tk = k + interaction_list[b][ui].shift[2];
    int tb = interaction_list[b][ui].shift[3];
    int tl = index(ti, tj, tk, tb);
    blenv <<= 2;
    blenv |= (mcL[tl]+1) & 0x3;
    int ci = (ui+1)%4;
    bint_compare[ci] <<= 2;
    bint_compare[ci] |= (mcL[tl]+1) & 0x3;
  }*/
  // faster
  /*
  array<uint64_t, 4> ilenv = {0, 0, 0, 0};
  ilenv[0] |= (mcL[l]+1) & 0x3;
  int ti, tj, tk, tb, tl, ci;
  for(int ui=0; ui<interaction_list[b].size(); ui++){
    ti = i + interaction_list[b][ui].shift[0];
    tj = j + interaction_list[b][ui].shift[1];
    tk = k + interaction_list[b][ui].shift[2];
    tb = interaction_list[b][ui].shift[3];
    tl = index(ti, tj, tk, tb);
    ci = ui%ilenv.size();
    ilenv[ci] <<= 2;
    ilenv[ci] |= (mcL[tl]+1) & 0x3;
  }*/
  // using local variables and unrolling the loops, fastest
  int ti, tj, tk, tb, tl, uj;
  //uint64_t bt1=0, bt2=0, bt3=0, bt4=0;
  uint64_t bt1=0, bt2=0, bt3=0, bt4=0, bt5=0;
  bt1 |= (mcL[l]+1) & 0x3;
  //for(int ui=0; ui+3<interaction_list[b].size(); ui+=4){
  for(int ui=0; ui+4<interaction_list[b].size(); ui+=5){
    ti = i + interaction_list[b][ui].shift[0];
    tj = j + interaction_list[b][ui].shift[1];
    tk = k + interaction_list[b][ui].shift[2];
    tb = interaction_list[b][ui].shift[3];
    tl = index(ti, tj, tk, tb);
    bt2 <<= 2;
    bt2 |= (mcL[tl]+1) & 0x3;
    ti = i + interaction_list[b][ui+1].shift[0];
    tj = j + interaction_list[b][ui+1].shift[1];
    tk = k + interaction_list[b][ui+1].shift[2];
    tb = interaction_list[b][ui+1].shift[3];
    tl = index(ti, tj, tk, tb);
    bt3 <<= 2;
    bt3 |= (mcL[tl]+1) & 0x3;
    ti = i + interaction_list[b][ui+2].shift[0];
    tj = j + interaction_list[b][ui+2].shift[1];
    tk = k + interaction_list[b][ui+2].shift[2];
    tb = interaction_list[b][ui+2].shift[3];
    tl = index(ti, tj, tk, tb);
    bt4 <<= 2;
    bt4 |= (mcL[tl]+1) & 0x3;
    ti = i + interaction_list[b][ui+3].shift[0];
    tj = j + interaction_list[b][ui+3].shift[1];
    tk = k + interaction_list[b][ui+3].shift[2];
    tb = interaction_list[b][ui+3].shift[3];
    tl = index(ti, tj, tk, tb);
    bt5 <<= 2;
    bt5 |= (mcL[tl]+1) & 0x3;
    ti = i + interaction_list[b][ui+4].shift[0];
    tj = j + interaction_list[b][ui+4].shift[1];
    tk = k + interaction_list[b][ui+4].shift[2];
    tb = interaction_list[b][ui+4].shift[3];
    tl = index(ti, tj, tk, tb);
    bt1 <<= 2;
    bt1 |= (mcL[tl]+1) & 0x3;
    uj = ui; // for the leftover
  }
  //uj += 4;
  uj += 5;
  if(uj<interaction_list[b].size()){
    ti = i + interaction_list[b][uj].shift[0];
    tj = j + interaction_list[b][uj].shift[1];
    tk = k + interaction_list[b][uj].shift[2];
    tb = interaction_list[b][uj].shift[3];
    tl = index(ti, tj, tk, tb);
    bt2 <<= 2;
    bt2 |= (mcL[tl]+1) & 0x3;
  }
  if(uj+1<interaction_list[b].size()){
    ti = i + interaction_list[b][uj+1].shift[0];
    tj = j + interaction_list[b][uj+1].shift[1];
    tk = k + interaction_list[b][uj+1].shift[2];
    tb = interaction_list[b][uj+1].shift[3];
    tl = index(ti, tj, tk, tb);
    bt3 <<= 2;
    bt3 |= (mcL[tl]+1) & 0x3;
  }
  if(uj+2<interaction_list[b].size()){
    ti = i + interaction_list[b][uj+2].shift[0];
    tj = j + interaction_list[b][uj+2].shift[1];
    tk = k + interaction_list[b][uj+2].shift[2];
    tb = interaction_list[b][uj+2].shift[3];
    tl = index(ti, tj, tk, tb);
    bt4 <<= 2;
    bt4 |= (mcL[tl]+1) & 0x3;
  }
  if(uj+3<interaction_list[b].size()){
    ti = i + interaction_list[b][uj+3].shift[0];
    tj = j + interaction_list[b][uj+3].shift[1];
    tk = k + interaction_list[b][uj+3].shift[2];
    tb = interaction_list[b][uj+3].shift[3];
    tl = index(ti, tj, tk, tb);
    bt5 <<= 2;
    bt5 |= (mcL[tl]+1) & 0x3;
  }
  //array<uint64_t, 4> ilenv;
  array<uint64_t, 5> ilenv;
  ilenv[0]=bt1; ilenv[1]=bt2; ilenv[2]=bt3; ilenv[3]=bt4; ilenv[4]=bt5;
  // xph warning: blenv/ilenv should be large enough to include all the possible local environment
  //              the lenv length depends on the number of non-zero ecis. 
  //const int nbytes = 28; // 28=224/8
  //const int nbytes = 32; // 32=64*4/8
  const int nbytes = 40; // 40=64*5/8
  uint32_t myseed = 0;
  uint32_t hi = XXHash32::hash(&ilenv, nbytes, myseed) % htcap[b]; // the hash table size is htcap=2^ht_size.
                                                                // the hash function returns a 32 bit int.
  //uint32_t hi = fnv1a(&blenv, sizeof(blenv)) >> (32-ht_size); // the hash table size is 2^ht_size.
  
  double energy;
  uint32_t  hj = hi;
  while(1){
    if(energy_map[b][hj].second > 999){
      htload[b]++;
      energy = pointenergy(i, j, k, b);
      //cout << "energy: "<< energy << endl;
      //energy_map[b][hj].first = blenv;
      energy_map[b][hj].first = ilenv;
      energy_map[b][hj].second = energy;
      //if(hj!=hi){htncol[b]++;}
      if(hj-hi>1){
        //cout << "================= ";
        cout << "collisions at " << hi << " : " << hj-hi << endl;
      }
      break;
    //} else if(energy_map[b][hj].first == blenv ){
    } else if(energy_map[b][hj].first == ilenv ){
      energy = energy_map[b][hj].second;
      //cout << "repeated local env" << endl;
      break;
    }
    hj++;
    if(hj >= energy_map[b].size()) hj=hj%htcap[b];
  }
  return energy;
}

//************************************************************
// xph: routines for operations of Binary Index Tree (Fenwick Tree)
// http://www.geeksforgeeks.org/binary-indexed-tree-or-fenwick-tree-2/
/*            n  --> No. of elements present in input array.   
    BITree[0..n] --> Array that represents Binary Indexed Tree.
    arr[0..n-1]  --> Input array for whic prefix sum is evaluated. */
 
// Returns sum of arr[0..index]. This function assumes
// that the array is preprocessed and partial sums of
// array elements are stored in BITree[].
double Monte_Carlo::getSum(double BITree[], int index)
{
    double sum = 0.0; // Iniialize result
 
    // index in BITree[] is 1 more than the index in arr[]
    index = index + 1;
 
    // Traverse ancestors of BITree[index]
    while (index>0)
    {
        // Add current element of BITree to sum
        sum += BITree[index];
 
        // Move index to parent node in getSum View
        index -= index & (-index);
    }
    return sum;
}
 
// Updates a node in Binary Index Tree (BITree) at given index
// in BITree.  The given value 'val' is added to BITree[i] and 
// all of its ancestors in tree.
void Monte_Carlo::updateBIT(double BITree[], int n, int index, double val)
{
    // index in BITree[] is 1 more than the index in arr[]
    index = index + 1;
 
    // Traverse all ancestors and add 'val'
    while (index <= n)
    {
       // Add 'val' to current node of BI Tree
       BITree[index] += val;
 
       // Update index to that of parent in update View
       index += index & (-index);
    }
}


//************************************************************

void Monte_Carlo::grand_canonical(double beta, chempot mu, int n_pass, int n_equil_pass){
  fluctuation uncorr_susc;
  uncorr_susc.initialize(conc);

  if(n_pass <=n_equil_pass){
    cout << "Npass must be larger than Nequil\n";
    cout << "Quitting grand_canonical()\n";
    exit(1);
  }

  //initialize all the thermodynamic averages to zero
  AVenergy=0.0;
  heatcap=0.0;
  AVconc.set_zero();
  AVnum_atoms.set_zero();
  AVsublat_conc.set_zero();
  AVSusc.set_zero();
  flipfreq=0.0;

  if(corr_flag){
    for(int n=0; n<AVcorr.size(); n++){
      AVcorr[n]=0.0;
    }
  }

  //copy mu into the basis objects and update the flip and dmu
  update_mu(mu);

  //double grand_energy=calc_grand_canonical_energy(mu);

  //cout << "grand canonical energy = " << grand_energy/nuc << "\n";

  // initilize the rates
  bitset<nmoves> allmove; //label all moves as update needed
  allmove.set();
  for(int l=0; l<nmcL; l++){
    update_rate_table(l, beta, allmove);
  }

  // xph: kmc runs
  // file for saving structrues
  string track;
  for(int i_m=0; i_m<mu.m.size(); i_m++){
    for(int j_m=0; j_m<mu.m[i_m].size()-1; j_m++){
      if(abs(mu.m[i_m][j_m])>1e-6){
        string ttrack;
        track.append(".");
        int_to_string(int(round(mu.m[i_m][j_m]*1000)), ttrack, 10);
        track.append(ttrack);
      }
    }
  }
  ofstream trajectory;
  string trajectory_file = "kmctraj";
  trajectory_file.append(track);
  trajectory_file.append(".xyz");
  trajectory.open(trajectory_file.c_str());
  ofstream numLitime; // file for saving numLi-time
  string numLitime_file = "numLi_";
  numLitime_file.append(track);
  numLitime_file.append(".dat");
  numLitime.open(numLitime_file.c_str());
  // todo: read n_writeout from input file
  int n_writeout = 5000000;
  int n_count = 1000;
  int numLi=100, numLi_old=0; // track number of Li left
  double time_old = 0.0; // track time change

  double time = 0.0;
  // rateTable.size might be larger than the number of grid sites because of reservoir
  //vector<double> crate; //cumulated rate in 1d vector
                        // crate[pi] is the index to rateTable[l][m] index
  vector<int> ctol; //ctol[pi] == l
  vector<int> ctom; //ctom[pi] == m
  vector<vector<int> > lmtoc; //lmtoc[l][m] == pi
  lmtoc.resize(nmcL);
  int pi=0;
  for(int l=0; l<nmcL; l++){
    int mtot = rateTable[l].size();
    lmtoc[l].resize(mtot);
    for(int mj=0; mj<mtot; mj++){
      ctol.push_back(l);
      ctom.push_back(mj);
      lmtoc[l][mj] = pi;
      pi++;
    }
  }
  //crate.resize(ctol.size());

  // save the cumulated rate in a Binary Indexed Tree.
  // Constructs a Binary Indexed Tree for given array of size nk.
  // Create and initialize BITree[] as 0
  int nk = ctol.size(); // total number of rates
  double *BITree = new double[nk+1];
  for (int i=1; i<=nk; i++)
    BITree[i] = 0.0;
 
  // get the cumulated rate array
  for(int l=0; l<nmcL; l++){
    for(int mj=0; mj<rateTable[l].size(); mj++){
      int pi=lmtoc[l][mj];
      // Store the actual values in BITree[] using update()
      updateBIT(BITree, nk, pi, rateTable[l][mj]);
    }
  }
 
  // xph: time the loop
  clock_t t;
  t = clock();
  for(int n=0; n<n_pass; n++){
    // crate is replaced by BITree.
    double tot_rate = getSum(BITree, nk-1);
    /*tot_rate=0.0;
    int pi=0;
    // get the cumulated rate array
    for(int l=0; l<nmcL; l++){
      for(int mj=0; mj<rateTable[l].size(); mj++){
        tot_rate += rateTable[l][mj];
        crate[pi] = tot_rate;
        pi++;
      }
    }*/

    // one random number for the escaping time
    time = time -log(ran0(idum))/tot_rate;

    // another random number to pick a event from the rate table
    double ran=ran0(idum)*tot_rate;
    int ir=ctol.size()-1;
    int il=-1;
    while(ir-il > 1){
      double interv=ir-il;
      int mid=int(ceil(interv/2.0))+il;

      //if(crate[mid] > ran) ir=mid;
      if(getSum(BITree, mid) > ran) ir=mid;
      else il=mid;
    }

    // follow the event to update mcL;
    int ls = ctol[ir];
    int ms = ctom[ir];
    int i=ltoi[ls];
    int j=ltoj[ls];
    int k=ltok[ls];
    int b=ltob[ls];

    // construct a map of {sites:moves} that list all the updates needed
    // moves are stored as a bitset with nmoves bits.
    map<int,bitset<nmoves> > update_site_moves;
    // construct update_site_moves for initial site of the hop
    for(int ui=0; ui<partial_update_list[b].size(); ui++){
      bitset<nmoves> movetags(partial_update_list[b][ui].name);
      int ti = i + partial_update_list[b][ui].shift[0];
      int tj = j + partial_update_list[b][ui].shift[1];
      int tk = k + partial_update_list[b][ui].shift[2];
      int tb = partial_update_list[b][ui].shift[3];
      int tl = index(ti, tj, tk, tb);
      update_site_moves[tl] = movetags;
    }
    for(int ui=0; ui<interaction_list[b].size(); ui++){
      int ti = i + interaction_list[b][ui].shift[0];
      int tj = j + interaction_list[b][ui].shift[1];
      int tk = k + interaction_list[b][ui].shift[2];
      int tb = interaction_list[b][ui].shift[3];
      int tl = index(ti, tj, tk, tb);
      update_site_moves[tl] = allmove;
    }
    if(ms<nmoves){
      //canonical move
      int b_2 = movemap[b][ms][0]; // basis of the destination site
      int i_2 = i + movemap[b][ms][1]; // cell index of the destination
      int j_2 = j + movemap[b][ms][2];
      int k_2 = k + movemap[b][ms][3];
      int l_2=index(i_2,j_2,k_2,b_2);
      if(b_2 == -1){
        cout << "BUG: picked an undefined canonical move";
        continue;
      }
      // exchange spins of the two selected sites.
      int tspin = mcL[ls];
      mcL[ls]   = mcL[l_2];
      mcL[l_2]  = tspin;
      
      // for final site of the hop
      for(int ui=0; ui<partial_update_list[b_2].size(); ui++){
        bitset<nmoves> movetags(partial_update_list[b_2][ui].name);
        int ti = i_2 + partial_update_list[b_2][ui].shift[0];
        int tj = j_2 + partial_update_list[b_2][ui].shift[1];
        int tk = k_2 + partial_update_list[b_2][ui].shift[2];
        int tb = partial_update_list[b_2][ui].shift[3];
        int tl = index(ti, tj, tk, tb);
        if(update_site_moves.count(tl) > 0){ 
          update_site_moves[tl] |= movetags; //OR, assign
        } else {
          update_site_moves[tl] = movetags; 
        }
      }
      for(int ui=0; ui<interaction_list[b_2].size(); ui++){
        int ti = i_2 + interaction_list[b_2][ui].shift[0];
        int tj = j_2 + interaction_list[b_2][ui].shift[1];
        int tk = k_2 + interaction_list[b_2][ui].shift[2];
        int tb = interaction_list[b_2][ui].shift[3];
        int tl = index(ti, tj, tk, tb);
        // if the key "tl" exists, no mater what movetags it has, replace the tags with allmove
        // so this part must be placed after checking the partial_update_list[b]
        update_site_moves[tl] = allmove;
      }
    } else {
      // Ni_grand: skip Ni grand move when Li% > 0.5
      if(numLi > nmcL/12 and ms-nmoves == 1) continue;
      // grandcanonical move
      // note:keep this part consistent with update_rate_table
      //int Lispin  = basis[b].get_spin("Li");
      //int Nispin  = basis[b].get_spin("Ni");
      //int Vacspin = basis[b].get_spin("Vac");
      int allowspin = (ms-nmoves == 0) ? Lispin: Nispin;
      if(mcL[ls] != allowspin and mcL[ls] != Vacspin){
        cout << "BUG: picked an undefined grandcanonical move";
        continue;
      }

      // determine index of current occupant at site l
      int co=basis[b].iflip(mcL[ls]);
      // determine index of next step occupant at site l
      int f=-1;
      for(int ti=0; ti<basis[b].flip[co].size(); ti++){
        int newspin=basis[b].flip[co][ti];
        if(newspin == allowspin or newspin == Vacspin){
          f=ti;
          break;
        }
      }
      if(f == -1){
        cout << "BUG: final spin in grand move is not allowed";
        continue;
      }
      mcL[ls]=basis[b].flip[co][f];
    }

    // update rateTable, only entries that are affected by the above move
    for(map<int,bitset<nmoves>>::iterator it=update_site_moves.begin(); it!=update_site_moves.end(); ++it){
      int l = it->first;
      vector<double> rates_old(rateTable[l]);
      update_rate_table(l, beta, it->second);
      // update accumulated rate in BITree
      for(int mj=0; mj<rates_old.size(); mj++){
        // no need to update
        if(mj < nmoves and not it->second.test(mj)) continue;
        //if(mj >= nmoves and it->second.count() < it->second.size()) continue; 
        updateBIT(BITree, nk, lmtoc[l][mj], rateTable[l][mj]-rates_old[mj]);
      }
    }
    
    // print time and number of Li
    if(n % n_count == 0){
      numLi=0;
      for(int tl=0; tl<nmcL; tl++){
        if(mcL[tl] == Lispin){
          numLi++;
        }
      }
      cout << "time: " << time << " "<< numLi << endl;
      cout << "load: " << htload[0]/(double)htcap[0] << " | " << htload[1]/(double)htcap[1] << endl;
      //cout << "collision ratio: " << htncol[0]/(double)htload[0] << " | " << htncol[1]/(double)htload[1] << endl;
      numLitime << time << " " << numLi << endl;
      // v1.6: constant voltammetric scan rate
      if(n==1){
        numLi_old = numLi;
      }else if(time >= 10){
        break;
      }
    }
    // write out structure
    if(n % n_writeout == 0){
      write_monte_xyz(trajectory);
    }
    /*if(n % (n_writeout) == 0){
      //if(abs(numLi - numLi_old)<2 and n>1) break;
      // constant current
      //if(abs(numLi - numLi_old)/(time - time_old) < 2 and n>1) break;
      time_old  = time;
      numLi_old = numLi;
    }*/
  }
  t = clock() - t;
  cout << "time for one mu:" << ((float)t)/CLOCKS_PER_SEC << endl;
  double current = (numLi - numLi_old)/time;
  cout << "current: " << current << endl;
  trajectory.close();
  numLitime.close();


  //---------------------------------------------
  // original grand canonical MC part
  /*
  for(int n=0; n<n_pass; n++){
    //pick nmcL lattice sites at random
    for(int nn=0; nn<nmcL; nn++){
      i=int(di*ran0(idum));
      j=int(dj*ran0(idum));
      k=int(dk*ran0(idum));
      b=int(db*ran0(idum));
      l=index(i,j,k,b);


      //determine index of current occupant at site l
      int co=basis[b].iflip(mcL[l]);

      //pick a flip event
      int f=int(ran0(idum)*basis[b].flip[co].size());

      int tspin=mcL[l];

      //get point energy before the spin flip
      double en_before=pointenergy(i,j,k,b)-basis[b].mu[co];

      //get point energy after the spin flip
      mcL[l]=basis[b].flip[co][f];
      int no=basis[b].iflip(mcL[l]);
      double en_after=pointenergy(i,j,k,b)-basis[b].mu[no];

      double delta_energy;
      delta_energy=en_after-en_before;

      if(delta_energy < 0.0 || exp(-delta_energy*beta) >=ran0(idum)){
        flipfreq++;
        grand_energy=grand_energy+delta_energy;
      }
      else{
        mcL[l]=tspin;
      }
    }

// AU
    cout << n << "  " << grand_energy/nuc << " EQUI \n";
// END AU

    if(n >= n_equil_pass){
      if(corr_flag){
	//Update average correlations
	for(i=0; i<di; i++){
	  for(j=0; j<dj; j++){
	    for(k=0; k<dk; k++){
	      for(b=0; b<db; b++){
		pointcorr(i, j, k, b);
	      }
	    }
	  }
	}
      }
      AVenergy=AVenergy+grand_energy;
      heatcap=heatcap+(grand_energy*grand_energy);
      calc_num_atoms();
      calc_concentration();
      calc_sublat_concentration();
      AVconc.increment(conc);
      AVsublat_conc.increment(sublat_conc);
      AVnum_atoms.increment(num_atoms);
      Susc.evaluate(num_atoms);
      AVSusc.increment(Susc);
    }

  }
  */
//----------------------------------------------------------- 

  AVenergy=AVenergy/(n_pass-n_equil_pass);
  if(corr_flag){
    for(int n=0; n<AVcorr.size(); n++){
      AVcorr[n]=AVcorr[n]/(di*dj*dk*(n_pass-n_equil_pass));
    }
  }
  heatcap=heatcap/(n_pass-n_equil_pass);
  heatcap=(heatcap-(AVenergy*AVenergy))*(beta*beta)*kb;
  AVconc.normalize(n_pass-n_equil_pass);
  AVsublat_conc.normalize(n_pass-n_equil_pass);
  AVnum_atoms.normalize(n_pass-n_equil_pass);
  uncorr_susc.evaluate(AVnum_atoms);
  AVSusc.normalize(n_pass-n_equil_pass);
  AVSusc.decrement(uncorr_susc);
  AVSusc.normalize(1.0/beta);
  flipfreq=flipfreq/(n_pass*nmcL);

}

//************************************************************


//************************************************************
//Added by Aziz : Beginning
//************************************************************

void Monte_Carlo::canonical_single_species(
  double beta, int n_pass, int n_equil_pass, int n_writeout){

  // This subroutine performs canonical Monte Carlo simulations in the
  // specific case of a single species cluster expansion

  int i_1,j_1,k_1,b_1,l_1; //Position indices for site 1
  int i_2,j_2,k_2,b_2,l_2; //Position indices for site 2

  fluctuation uncorr_susc;
  uncorr_susc.initialize(conc);

  if(n_pass <=n_equil_pass){
    cout << "Npass must be larger than Nequil\n";
    cout << "Quitting canonical_single_species \n";
    exit(1);
  }

  //initialize all the thermodynamic averages to zero
  AVenergy=0.0;
  heatcap=0.0;
  AVconc.set_zero();
  AVnum_atoms.set_zero();
  AVsublat_conc.set_zero();
  AVSusc.set_zero();
  flipfreq=0.0;

  if(corr_flag){
    for(int n=0; n<AVcorr.size(); n++){
      AVcorr[n]=0.0;
    }
  }

  //Calculating the total energy
  double energy;
  calc_energy(energy);

  //Outputting the value to the screen
  cout << "energy = " << energy/nuc << "\n";

  bool right_combination=false; // This variable tells us if a valid
                                // electron/hole combination has been
                                // found

  ofstream trajectory;
  string trajectory_file = "mctrajec.xyz";
  trajectory.open(trajectory_file.c_str());

  for(int n=0; n<n_pass; n++){
    // n_pass passes are made. For each pass, pick nmcL electron/hole
    // pairs at random

    for(int nn=0; nn<nmcL; nn++){

      // Loop until an electron/hole pair is selected. The boolean
      // variable "right_combination" tells us if a valid combination
      // has been identified.

      right_combination = false;

      while (right_combination==false) {
        i_1=int(di*ran0(idum));
        j_1=int(dj*ran0(idum));
        k_1=int(dk*ran0(idum));
        b_1=int(db*ran0(idum));
        l_1=index(i_1,j_1,k_1,b_1);

        i_2=int(di*ran0(idum));
        j_2=int(dj*ran0(idum));
        k_2=int(dk*ran0(idum));
        b_2=int(db*ran0(idum));
        l_2=index(i_2,j_2,k_2,b_2);

        //Testing if a valid combination (Species-Va) has been identified
	    if ((mcL[l_2]==1 && mcL[l_1]==-1) ){
	      right_combination=true;
        }

	    if ((mcL[l_2]==-1 && mcL[l_1]==1) ){
	      right_combination=true;
        }
      }

	  //Storing the initial spin states
      int tspin_2=mcL[l_2];
	  int tspin_1=mcL[l_1];

	  // To calculate the difference in energy due to the 2 flips, we
	  // first perform the flip on site 2 and then the flip on site
	  // 1. We calculate the pointenergy of each site, before and after
	  // the flip.

	  //Site 2 flip
      //get point energy before the flip of site 2
      double en_before_flip_2=pointenergy(i_2,j_2,k_2,b_2);

      //get point energy after the flip of site 2
      mcL[l_2]=-mcL[l_2];
      double en_after_flip_2=pointenergy(i_2,j_2,k_2,b_2);

      double delta_energy_flip_2;
      delta_energy_flip_2=en_after_flip_2-en_before_flip_2;

	  //Site 1 flip
      //get point energy before the flip of site 1
	  double en_before_flip_1=pointenergy(i_1,j_1,k_1,b_1);

      //Get point energy after the flip of site 1
      mcL[l_1]=-mcL[l_1];
	  double en_after_flip_1=pointenergy(i_1,j_1,k_1,b_1);

      double delta_energy_flip_1;
      delta_energy_flip_1=en_after_flip_1-en_before_flip_1;

	  //Calculating the total change in energy
	  double delta_energy=delta_energy_flip_2+delta_energy_flip_1;

      if (delta_energy < 0.0 || exp(-delta_energy*beta) >=ran0(idum)) {
        flipfreq++;
        energy=energy+delta_energy;
      } else {
        mcL[l_2]=tspin_2;
	    mcL[l_1]=tspin_1;
      }

    } // for nn (nmcL flips)

	// Calculating the correlations
    // (Note : We have to check if these correlation things still apply)

    if(n >= n_equil_pass){
      if(corr_flag){
        //Update average correlations
        for(int i=0; i<di; i++){
          for(int j=0; j<dj; j++){
            for(int k=0; k<dk; k++){
              for(int b=0; b<db; b++){
                pointcorr(i, j, k, b);
              }
            }
          }
        }
      }
      AVenergy=AVenergy+energy;
      heatcap=heatcap+(energy*energy);
      calc_num_atoms();
      calc_concentration();
      calc_sublat_concentration();
      AVconc.increment(conc);
      AVsublat_conc.increment(sublat_conc);
      AVnum_atoms.increment(num_atoms);
      Susc.evaluate(num_atoms);
      AVSusc.increment(Susc);

      // write out structure
      if (n % n_writeout == 0) {
        write_monte_xyz(trajectory);
      }

    } // n >= n_equil_pass --> production steps

  } // for n passes (MC steps)

  AVenergy=AVenergy/(n_pass-n_equil_pass);
  if(corr_flag){
    for(int n=0; n<AVcorr.size(); n++){
      AVcorr[n]=AVcorr[n]/(di*dj*dk*(n_pass-n_equil_pass));
    }
  }
  heatcap=heatcap/(n_pass-n_equil_pass);
  heatcap=(heatcap-(AVenergy*AVenergy))*(beta*beta)*kb;
  AVconc.normalize(n_pass-n_equil_pass);
  AVsublat_conc.normalize(n_pass-n_equil_pass);
  AVnum_atoms.normalize(n_pass-n_equil_pass);
  uncorr_susc.evaluate(AVnum_atoms);
  AVSusc.normalize(n_pass-n_equil_pass);
  AVSusc.decrement(uncorr_susc);
  AVSusc.normalize(1.0/beta);
  flipfreq=flipfreq/(n_pass*nmcL);

  trajectory.close();

} // canonical_single_species

//************************************************************
//Added by Aziz : End
//************************************************************

/*--------------------------------------------------------------------
                                BEGIN AU
       simulated annealing canonical MC for a single CE species
  --------------------------------------------------------------------*/

void Monte_Carlo::anneal_single_species(
  double T_init, double T_final, int n_pass){

  /*
    This method performs a simulated annealing simulation from initial
    temperature T_init to final temperature T_final in n_pass steps.  It
    is restricted to a single CE species and will keep its concentration
    fixed. The code is based on `canonical_single_species'.

    2014-01-10 Alexander Urban (AU)
  */

  /*
    i, j, k, b, l       position indices for sites 1 and 2
    right_combination   right combination of occupied & vacant sites?
    energy              total energy
    T                   temperature
    dT                  temperature change per MC step
    beta                1/(kb*T)
  */
  int    i_1,j_1,k_1,b_1,l_1;
  int    i_2,j_2,k_2,b_2,l_2;
  bool   right_combination;
  double energy;
  double T;
  double dT;
  double beta;

  double en_before_flip_1, en_before_flip_2;
  double en_after_flip_1, en_after_flip_2;
  double delta_energy;

  flipfreq = 0.0;
  calc_energy(energy);

  // temperature change per MC step
  dT = (T_final - T_init)/double(n_pass);
  T  = T_init;

  printf("%7d  %8.2f  %15.8e  *ANNEAL*\n", 0, T, energy/nuc);

  for(int n=0; n<n_pass; n++){
    beta = 1.0/(kb*T);
    for(int nn=0; nn<nmcL; nn++){
      // n_pass passes are made. For each pass, pick nmcL electron/hole
      // pairs at random

      right_combination = false;
      while (right_combination == false) {
        /* Loop until an electron/hole pair is selected. The boolean
           variable "right_combination" tells us if a valid combination
           has been identified. */
        i_1 = int(di*ran0(idum));
        j_1 = int(dj*ran0(idum));
        k_1 = int(dk*ran0(idum));
        b_1 = int(db*ran0(idum));
        l_1 = index(i_1,j_1,k_1,b_1);
        i_2 = int(di*ran0(idum));
        j_2 = int(dj*ran0(idum));
        k_2 = int(dk*ran0(idum));
        b_2 = int(db*ran0(idum));
        l_2 = index(i_2,j_2,k_2,b_2);
        // valid combination (species & vacancy) ?
	    if (mcL[l_2] + mcL[l_1] == 0) { right_combination = true; }
      }

	  //Storing the initial spin states
      int tspin_2 = mcL[l_2];
	  int tspin_1 = mcL[l_1];

	  /*
        To calculate the difference in energy due to the 2 flips, we
        first perform the flip on site 2 and then the flip on site
        1. We calculate the pointenergy of each site, before and after
        the flip.
      */

	  // site 2 flip:
      // get point energy before/after the flip of site 2
      en_before_flip_2 = pointenergy(i_2,j_2,k_2,b_2);
      mcL[l_2]         = -mcL[l_2];
      en_after_flip_2  = pointenergy(i_2,j_2,k_2,b_2);

      // energy change upon flip of site 2
      delta_energy = en_after_flip_2 - en_before_flip_2;

	  // site 1 flip: get point energy before/after the flip of site 1
	  en_before_flip_1 = pointenergy(i_1,j_1,k_1,b_1);
      mcL[l_1]         = -mcL[l_1];
	  en_after_flip_1  = pointenergy(i_1,j_1,k_1,b_1);

      // energy change upon flip of site 1
      delta_energy += en_after_flip_1 - en_before_flip_1;

      if (delta_energy < 0.0 || exp(-delta_energy*beta) >=ran0(idum)) {
        flipfreq++;
        energy = energy + delta_energy;
      } else { // reverse flips
        mcL[l_2] = tspin_2;
	    mcL[l_1] = tspin_1;
      }

    } // for nn (nmcL flips)
    T += dT;
    printf("%7d  %8.2f  %15.8e  *ANNEAL*\n", n, T, energy/nuc);
  } // for n passes (MC steps)

  flipfreq = flipfreq/(n_pass*nmcL);

} // anneal_single_species
// END AU

/*--------------------------------------------------------------------*/

// Code Edited - function added by John Thomas
double Monte_Carlo::lte(double beta, chempot mu){

  int l;

  update_mu(mu);
  double phi_expanded=calc_grand_canonical_energy(mu);

  for(int b=0; b<db; b++){
    for(int i=0; i<di; i++){
      for(int j=0; j<dj; j++){
	for(int k=0; k<dk; k++){

	  l=index(i,j,k,b);

	  int co=basis[b].iflip(mcL[l]);

	  for(int f=0; f<basis[b].flip[co].size(); f++){

	    int tspin=mcL[l];
	    double en_before=pointenergy(i,j,k,b)-basis[b].mu[co];
	    //cout << "en_before = "<< en_before << "\n";

	    mcL[l]=basis[b].flip[co][f];
	    int no=basis[b].iflip(mcL[l]);
	    double en_after=pointenergy(i,j,k,b)-basis[b].mu[no];
	    //cout << "en_after = "<< en_after << "\n";

	    double delta_energy;
	    delta_energy=en_after-en_before;
	    mcL[l]=tspin;

	    if(delta_energy<0){
	      cout << "Configuration is not a ground state at current chemical potential.  Please re-initialize with correct ground state. \n";
	      exit(1);
	    }
	    phi_expanded=phi_expanded-exp(-delta_energy*beta)/beta;
	  }
	}
      }
    }
  }

  phi_expanded=phi_expanded/nuc;
  cout << "Finished Low Temperature Expansion.  Free energy at initial temperature is " << phi_expanded << "\n";
  return phi_expanded;
}
//\end edited code

//************************************************************

void Monte_Carlo::initialize_kmc(){

  //for each basis site, find the corresponding multiplet of montiplet
  //push back the nearest neighbor clusters into endpoints

  char name[2];
  name[0]='V';
  name[1]='a';



  //read in the hop clusters
  //identify final states and activated states
  //generate basiplet and then montiplet for each site
  //collect other information - shift vectors, hop vectors etc.

  string hop_file="hops";

  ifstream in;
  in.open(hop_file.c_str());
  if(!in){
    cout << hop_file << " cannot be read \n";
    cout << "not initializing kinetic Monte Carlo \n";
    return;
  }

  int nhops;
  vector<orbit> hoporb;
  char buff[200];
  in >> nhops;
  in.getline(buff,199);
  for(int i=0; i< nhops; i++){
    cluster tclus;
    int np;
    in >> np >> np;
    in.getline(buff,199);
    if(np != 2 && np != 3){
      cout << "hop cluster in " << hop_file << " is not legitimate \n";
      cout << "use only 2 or 3 point clusters \n";
      cout << "not initializing kinetic Monte Carlo \n";
      return;
    }

    tclus.readf(in,np);
    //for each point indicate whether they are part of the regular lattice or are an activated
    //state
    //the bit in atompos is used to indicate whether the point is regular or activated
    // regular: bit = 0 ; activated: bit = 1.
    //if there are only two sites in the hop, then all sites are regular
    //if there are three sites in the hop, then the middle one is the activated

    if(tclus.point.size() == 2){
      for(int j=0; j<2; j++){
	tclus.point[j].bit=0;
      }
    }

    if(tclus.point.size() == 3){
      for(int j=0; j<3; j++){
	tclus.point[j].bit=0;
      }
      tclus.point[1].bit=1;
    }

    orbit torb;
    torb.equiv.push_back(tclus);
    hoporb.push_back(torb);

  }

  in.close();

  //for each point in the hop clusters, we need to compare to basis[] and assign those points the allowed components etc


  for(int i=0; i< hoporb.size(); i++){
    for(int j=0; j< hoporb[i].equiv.size(); j++){
      for(int k=0; k<hoporb[i].equiv[j].point.size(); k++){
	bool mapped=false;
	for(int l=0; l< basis.size(); l++){
	  int trans[3];
	  if(compare(basis[l].fcoord,hoporb[i].equiv[j].point[k].fcoord,trans)){
	    hoporb[i].equiv[j].point[k].compon.clear();
	    for(int m=0; m < basis[l].compon.size(); m++){
	      hoporb[i].equiv[j].point[k].compon.push_back(basis[l].compon[m]);
	    }
	    mapped=true;
	  }
	}
	if(!mapped){
	  cout << " a point from the hop cluster was not mapped onto \n";
	  cout << "a basis site of your monte carlo system \n";
	}
      }
    }
  }




  //for each hop cluster we need to get the orbit using the prim symmetry operations

  for(int i=0; i< hoporb.size(); i++){
    get_equiv(hoporb[i],prim.factor_group);
  }

  //next get a montiplet type structure (hop clusters radiating out of each basis site using
  //get_ext_montiplet()
  //first we need to copy the vector of orbits into a multiplet

  multiplet hoptiplet;

  //fill up the empty and point slots in the hoptiplet first
  for(int i=0; i<2; i++){
    vector<orbit>torbvec;
    hoptiplet.orb.push_back(torbvec);
  }

  //check whether any of the hop clusters have two sites
  for(int i=0; i < hoporb.size(); i++){
    if(hoporb[i].equiv[0].point.size() == 2){
      if(hoptiplet.orb.size()<3){
	vector<orbit> torbvec;
	torbvec.push_back(hoporb[i]);
	hoptiplet.orb.push_back(torbvec);
      }
      else{
	hoptiplet.orb[2].push_back(hoporb[i]);
      }
    }
  }
  //if no hop clusters with two sites are present pushback an empty
  if(hoptiplet.orb.size()<3){
    vector<orbit>torbvec;
    hoptiplet.orb.push_back(torbvec);
  }

  //check whether any of the hop clusters have three sites
  for(int i=0; i < hoporb.size(); i++){
    if(hoporb[i].equiv[0].point.size() ==3){
      if(hoptiplet.orb.size()<4){
	vector<orbit> torbvec;
	torbvec.push_back(hoporb[i]);
	hoptiplet.orb.push_back(torbvec);
      }
      else{
	hoptiplet.orb[3].push_back(hoporb[i]);
      }
    }
  }

  //for each basis site get a montiplet

  vector<multiplet> montihoptiplet;

  generate_ext_monteclust(basis,hoptiplet,montihoptiplet);

  //now construct the hop class

  jumps.clear();

  if(basis.size() != montihoptiplet.size()){
    cout << "mismatch between the size of basis and montihoptiplet \n";
    cout << "not initializing kinetic Monte Carlo \n";
    return;
  }

  for(int i=0; i<basis.size(); i++){
    //make an empty vector of hops
    vector<hop> tjumpvec;
    jumps.push_back(tjumpvec);

    //determine whether this basis site is an endpoint of any hop - as opposed to only serving as an activated state
    bool endpoint=false;
    for(int np=2; np<montihoptiplet[i].orb.size(); np++){
      for(int n=0; n<montihoptiplet[i].orb[np].size(); n++){
	for(int l=0; l<montihoptiplet[i].orb[np][n].equiv[0].point.size(); l++){
	  if(compare(basis[i].fcoord,montihoptiplet[i].orb[np][n].equiv[0].point[l].fcoord) &&
	     compare(basis[i].compon,montihoptiplet[i].orb[np][n].equiv[0].point[l].compon)){
	    if(montihoptiplet[i].orb[np][n].equiv[0].point[l].bit == 0 ) endpoint = true;

	  }
	}
      }
    }

    if(endpoint){
      for(int np=2; np<montihoptiplet[i].orb.size(); np++){
	for(int n=0; n<montihoptiplet[i].orb[np].size(); n++){

	  //check whether this particular cluster contains the basis as an endpoint and not an activated state
	  bool endpoint2 = false;
	  for(int l=0; l<montihoptiplet[i].orb[np][n].equiv[0].point.size(); l++){
	    if(compare(basis[i].fcoord,montihoptiplet[i].orb[np][n].equiv[0].point[l].fcoord) &&
	       compare(basis[i].compon,montihoptiplet[i].orb[np][n].equiv[0].point[l].compon)){
	      if(montihoptiplet[i].orb[np][n].equiv[0].point[l].bit == 0 ) endpoint2 = true;
	    }
	  }

	  if(endpoint2){
	    hop tjump;
	    tjump.b=i;          // assign the basis index for this jump object
	    tjump.initial=basis[i];
	    tjump.vac_spin_init=basis[i].get_spin(name);

	    for(int ne=0; ne<montihoptiplet[i].orb[np][n].equiv.size(); ne++){

	      tjump.endpoints.push_back(montihoptiplet[i].orb[np][n].equiv[ne]);
	      tjump.endpoints[ne].get_cart(prim.FtoC);
	      for(int k=0; k<tjump.endpoints[ne].point.size(); k++){
		//for each of the neighbors, get the cartesian coordinates
		if(!compare(tjump.endpoints[ne].point[k],basis[i])){
		  if(tjump.endpoints[ne].point[k].bit == 0){
		    mc_index tfinal;
		    int tvac_spin;
		    vec tjump_vec;
		    double tleng;
		    for(int l=0; l<3; l++){
		      tjump_vec.fcoord[l]=tjump.endpoints[ne].point[k].fcoord[l]-basis[i].fcoord[l];
		      tjump_vec.ccoord[l]=tjump.endpoints[ne].point[k].ccoord[l]-basis[i].ccoord[l];
		      tjump_vec.ccoord[l]=tjump_vec.ccoord[l]*(1.0e-8);   //convert to cm
		      tjump_vec.frac_on=true;
		      tjump_vec.cart_on=true;
		    }
		    tleng=tjump_vec.calc_dist();
		    for(int l=0; l<4; l++){
		      tfinal.shift[l]=tjump.endpoints[ne].point[k].shift[l];
		    }
		    tvac_spin=tjump.endpoints[ne].point[k].get_spin(name);
		    tjump.jump_vec.push_back(tjump_vec);
		    tjump.jump_leng.push_back(tleng);
		    tjump.final.push_back(tfinal);
		    tjump.vac_spin.push_back(tvac_spin);
		  }
		  if(tjump.endpoints[ne].point[k].bit == 1){
		    mc_index tactivated;
		    for(int l=0; l<4; l++){
		      tactivated.shift[l]=tjump.endpoints[ne].point[k].shift[l];
		    }
		    tjump.activated.push_back(tactivated);
		  }
		}
	      }
	    }
	    bool clear=true;
	    tjump.get_reach(montiplet,clear,basis);
	    jumps[i].push_back(tjump);
	  }
	}
      }
    }
  }

  cout << "about to enter extend_reach() \n";

  extend_reach();

  cout << "just passed extend_reach() \n";

  for(int i=0; i < jumps.size(); i++){
    cout << "HOP INFO FOR BASIS SITE  i = " << i << "\n";
    for(int j=0; j < jumps[i].size(); j++){
      cout << " HOP j = " << j << "\n";
      jumps[i][j].print_hop_info(cout);
    }
  }

  //determine the length of the prob and cprob arrays
  np=0;
  for(int i=0; i<jumps.size(); i++){
    for(int j=0; j<jumps[i].size(); j++){
      np=np+jumps[i][j].endpoints.size();
    }
  }
  np=np*nuc;

  prob = new double[np];
  cprob = new double[np];
  ltosp = new int[nmcL];
  ptol = new int[np];
  ptoj = new int[np];  // probability to jump
  ptoh = new int[np];   //probability to hop


  if(basis.size() != jumps.size()){
    cout << "number of basis sites is not equal to number of jump vectors\n";
    cout << "some kind of error occurred \n";
    cout << "NOT CONTINUING WITH initialize_kmc()\n";
    return;
  }

  //figure out which basis sites serve as a regular site at least once (store that info as bit=0, otherwise bit=1)
  for(int i=0; i<basis.size(); i++){
    if(jumps[i].size() == 0) basis[i].bit =1;
    else basis[i].bit =0;
  }


  Rx = new double[nmcL];
  Ry = new double[nmcL];
  Rz = new double[nmcL];

  int p=0;
  for(int l=0; l<nmcL; l++){
    if(basis[ltob[l]].bit == 0){
      ltosp[l]=p;
      for(int j=0; j<jumps[ltob[l]].size(); j++){
	for(int h=0; h<jumps[ltob[l]][j].endpoints.size(); h++){
	  ptol[p]=l;
	  ptoj[p]=j;
	  ptoh[p]=h;
	  p++;
	}
      }
    }
  }

  cout << "WE HAVE FILLED UP ALL THE HOP ARRAYS \n";
  cout << " p = " << p << "\n";
  cout << " np = " << np << "\n";


  hop_leng=0.0;
  for(int i=0; i<jumps.size(); i++){
    for(int j=0; j<jumps[i].size(); j++){
      for(int k=0; k<jumps[i][j].jump_leng.size(); k++){
	if(jumps[i][j].jump_leng[k] > hop_leng) hop_leng = jumps[i][j].jump_leng[k];
      }
    }
  }

  cout << "The maximum hop distance is " << hop_leng << "\n";

  if(hop_leng == 0.0){
    cout << "the maximum hop length is zero \n";
    cout << "you will have problems in your kmc simulation \n";
    cout << "check your hop file \n";
  }


  R.initialize(conc);
  kinL.initialize(conc);
  Dtrace.initialize(conc);
  corrfac.initialize(conc);
  AVkinL.initialize(conc);
  AVDtrace.initialize(conc);
  AVcorrfac.initialize(conc);

}

//************************************************************
// goes through the jump structure and for every hop reach site, adds additional reach sites that include sites that
// each site in the existing reach can hop to

void Monte_Carlo::extend_reach(){

  for(int i=0; i<jumps.size(); i++){
    for(int j=0; j<jumps[i].size(); j++){
      for(int h=0; h < jumps[i][j].reach.size(); h++){
	int rs=jumps[i][j].reach[h].size();
	for(int r=0; r<rs; r++){
	  int b=jumps[i][j].reach[h][r].shift[3];
	  //for every hop from basis site b make the shift
	  for(int ht=0; ht < jumps[b].size(); ht++){
	    for(int hh=0; hh<jumps[b][ht].endpoints.size(); hh++){
	      mc_index treach;
	      treach.shift[0]=jumps[i][j].reach[h][r].shift[0]+jumps[b][ht].final[hh].shift[0];
	      treach.shift[1]=jumps[i][j].reach[h][r].shift[1]+jumps[b][ht].final[hh].shift[1];
	      treach.shift[2]=jumps[i][j].reach[h][r].shift[2]+jumps[b][ht].final[hh].shift[2];
	      treach.shift[3]=jumps[b][ht].final[hh].shift[3];

	      //check whether this treach point already exists in reach[h]
	      bool isnew=true;
	      for(int n=0; n<jumps[i][j].reach[h].size(); n++){
		if(compare(jumps[i][j].reach[h][n],treach)){
		  isnew=false;
		  break;
		}
	      }
	      if(isnew) jumps[i][j].reach[h].push_back(treach);


	    }
	  }
	}
      }
    }
  }


}


//************************************************************

void Monte_Carlo::get_hop_prob(int i, int j, int k, int b, double beta){
  double nu=1.0e13;

  if(jumps[b].size() == 0) return;

  int l=index(i,j,k,b);

  if(mcL[l] == -1){
    //  if(mcL[l] == jumps[b][0].vac_spin_init){

    int cumh=0;
    for(int ht=0; ht < jumps[b].size(); ht++){
      for(int h=0; h<jumps[b][ht].endpoints.size(); h++){
	int ii=i+jumps[b][ht].final[h].shift[0];
	int jj=j+jumps[b][ht].final[h].shift[1];
	int kk=k+jumps[b][ht].final[h].shift[2];
	int bb=jumps[b][ht].final[h].shift[3];

	int ll=index(ii,jj,kk,bb);
	int pp=ltosp[l]+cumh;
	cumh++;

	//check whether this endpoint is occupied

	if(mcL[ll] == 1){
	  //	if(mcL[ll] != jumps[b][ht].vac_spin[h]){
	  double barrier=calc_barrier(i,j,k,b,ii,jj,kk,bb,l,ll,ht,h);
	  if(barrier > 0) prob[pp]=nu*exp(-beta*barrier);
	  else prob[pp]=nu;
	}
	else{ //the endpoint h is not occupied
	  prob[pp]=0.0;
	}
      }
    }
  }
  else{  // the site is not vacant
    int cumh=0;
    for(int ht=0; ht < jumps[b].size(); ht++){
      for(int h=0; h<jumps[b][ht].endpoints.size(); h++){
	int pp=ltosp[l]+cumh;
	cumh++;
	prob[pp]=0.0;
      }
    }
  }
}


//************************************************************

double Monte_Carlo::calc_barrier(int i, int j, int k, int b, int ii, int jj, int kk, int bb, int l, int ll, int ht, int h){
  double barrier,penA1,penA2,penB1,penB2;
  double kra=0.28;

  if(jumps[b][ht].activated.size() > 0){   // intermediate activated state

    int iii=i+jumps[b][ht].activated[h].shift[0];
    int jjj=j+jumps[b][ht].activated[h].shift[1];
    int kkk=k+jumps[b][ht].activated[h].shift[2];
    int bbb=jumps[b][ht].activated[h].shift[3];

    int lll=index(iii,jjj,kkk,bbb);


    penA1=pointenergy(ii,jj,kk,bb);
    penB1=pointenergy(iii,jjj,kkk,bbb);

    mcL[ll]=-mcL[ll];
    mcL[lll]=-mcL[lll];
    penA2=pointenergy(ii,jj,kk,bb);
    penB2=pointenergy(iii,jjj,kkk,bbb);

    mcL[ll]=-mcL[ll];
    mcL[lll]=-mcL[lll];

    barrier=(penA2+penB2-penA1-penB1);
    return barrier;
  }
  else{   // no intermediate activated state
    penB1=pointenergy(i,j,k,b);
    penB2=pointenergy(ii,jj,kk,bb);

    mcL[l]=-mcL[l];
    mcL[ll]=-mcL[ll];

    penA1=pointenergy(i,j,k,b);
    penA2=pointenergy(ii,jj,kk,bb);

    mcL[l]=-mcL[l];
    mcL[ll]=-mcL[ll];


    barrier=0.5*(penA1+penA2-penB1-penB2)+kra;
    return barrier;

  }
}





//************************************************************

void Monte_Carlo::initialize_prob(double beta){

  for(int i=0; i<di; i++){
    for(int j=0; j<dj; j++){
      for(int k=0; k<dk; k++){
	for(int b=0; b<db; b++){
	  if(basis[b].bit == 0){
	    get_hop_prob(i,j,k,b,beta);
	  }
	}
      }
    }
  }
}


//************************************************************

int Monte_Carlo::pick_hop(){
  //get cummulative hop array
  //pick the interval in which ran0(idum) falls
  //the event that occurs is the index bounding the interval from above

  cprob[0]=prob[0];
  for(int p=1; p<np; p++) cprob[p]=cprob[p-1]+prob[p];

  tot_prob=cprob[np-1];

  double ran=ran0(idum)*tot_prob;

  int il=np-1;
  int mm=il;
  int ir=-1;

  while(il-ir > 1){
    double interv=il-ir;
    int mid=int(ceil(interv/2.0))+ir;

    if(cprob[mid] > ran) il=mid;
    else ir=mid;
  }

  return il;

}


//************************************************************

void Monte_Carlo::update_prob(int i, int j, int k, int b, int ht, int h, double beta){
  for(int n=0; n<jumps[b][ht].reach[h].size(); n++){
    int ii=i+jumps[b][ht].reach[h][n].shift[0];
    int jj=j+jumps[b][ht].reach[h][n].shift[1];
    int kk=k+jumps[b][ht].reach[h][n].shift[2];
    int bb=jumps[b][ht].reach[h][n].shift[3];
    get_hop_prob(ii,jj,kk,bb,beta);
  }
}




//************************************************************

void Monte_Carlo::kinetic(double beta, double n_pass, double n_equil_pass){

  //do some initializations of R-vectors and fluctuation variables
  for(int l=0; l<nmcL; l++){
    Rx[l]=0.0;
    Ry[l]=0.0;
    Rz[l]=0.0;
  }

  double tRx,tRy,tRz;
  double kmc_time=0.0;

  R.set_zero();
  kinL.set_zero();
  Dtrace.set_zero();
  corrfac.set_zero();
  num_hops.set_zero();
  AVkinL.set_zero();
  AVDtrace.set_zero();
  AVcorrfac.set_zero();

  calc_num_atoms();

  cout << "The number of atoms are \n";
  num_atoms.print_concentration(cout);
  cout << "\n";

  calc_concentration();


  initialize_prob(beta);


  for(int n=0; n<n_pass; n++){
    for(int nn=0; nn<nmcL; nn++){

      int p=pick_hop();

      //determine site of the hop and the hop for that site
      int l=ptol[p];
      int h=ptoh[p];
      int ht=ptoj[p];

      int i=ltoi[l];
      int j=ltoj[l];
      int k=ltok[l];
      int b=ltob[l];


      //determine the end point of the hop

      int ii=i+jumps[b][ht].final[h].shift[0];
      int jj=j+jumps[b][ht].final[h].shift[1];
      int kk=k+jumps[b][ht].final[h].shift[2];
      int bb=jumps[b][ht].final[h].shift[3];


      int ll=index(ii,jj,kk,bb);


      int temp=mcL[l];
      mcL[l]=mcL[ll];
      mcL[ll]=temp;

      tRx=Rx[l];
      tRy=Ry[l];
      tRz=Rz[l];

      Rx[l]=Rx[ll]-jumps[b][ht].jump_vec[h].ccoord[0];
      Ry[l]=Ry[ll]-jumps[b][ht].jump_vec[h].ccoord[1];
      Rz[l]=Rz[ll]-jumps[b][ht].jump_vec[h].ccoord[2];

      Rx[ll]=tRx+jumps[b][ht].jump_vec[h].ccoord[0];
      Ry[ll]=tRy+jumps[b][ht].jump_vec[h].ccoord[1];
      Rz[ll]=tRz+jumps[b][ht].jump_vec[h].ccoord[2];

      kmc_time=kmc_time-log(ran0(idum))/tot_prob;

      update_num_hops(l,ll,b,bb);

      update_prob(i,j,k,b,ht,h,beta);
    }


    if(n > n_equil_pass){

      collect_R();

      kinL.evaluate(R);
      Dtrace=R;
      corrfac=R;
      Dtrace.normalize(num_atoms);

      double norm1=kmc_time*6.0;
      double norm2=hop_leng*hop_leng;

      kinL.normalize(norm1);
      double size=nuc;
      kinL.normalize(size,num_atoms);
      Dtrace.normalize(norm1);
      corrfac.normalize(norm2);
      corrfac.normalize(num_hops);


      AVkinL.increment(kinL);
      AVDtrace.increment(Dtrace);
      AVcorrfac.increment(corrfac);

    }

  }


  double norm3=n_pass-n_equil_pass;

  AVkinL.normalize(norm3);
  AVDtrace.normalize(norm3);
  AVcorrfac.normalize(norm3);


}



//************************************************************

void Monte_Carlo::collect_R(){



  for(int i=0; i<R.spin.size(); i++){

    for(int j=0; j<R.spin[i].size(); j++){
      R.Rx[i][j]=0.0;
      R.Ry[i][j]=0.0;
      R.Rz[i][j]=0.0;
      R.R2[i][j]=0.0;
      for(int l=0; l<nmcL; l++){
	//if(basis[ltob[l]].basis_flag == '0'){
	if(basis[ltob[l]].bit == 0){
	  if(mcL[l] == R.spin[i][j]){
	    R.Rx[i][j]=R.Rx[i][j]+Rx[l];
	    R.Ry[i][j]=R.Ry[i][j]+Ry[l];
	    R.Rz[i][j]=R.Rz[i][j]+Rz[l];
	    R.R2[i][j]=R.R2[i][j]+Rx[l]*Rx[l]+Ry[l]*Ry[l]+Rz[l]*Rz[l];
	  }
	}
      }
    }
  }
}


//************************************************************

void hop::get_reach(vector<multiplet> montiplet, bool clear, vector<atompos> basis){

  if(clear) reach.clear();
  for(int i=0; i < endpoints.size(); i++){
    vector < mc_index > treach;
    for(int j=0; j < endpoints[i].point.size(); j++){
      for(int k=0; k< montiplet.size(); k++){
	if(montiplet[k].orb[1].size()>1 || montiplet[k].orb[1][0].equiv.size() > 1 || montiplet[k].orb[1][0].equiv[0].point.size() > 1){
	  cout << "we have an unanticipated montiplet structure\n";
	  cout << "leaving get_reach and reach vector is not constructed \n";
	  cout << "expect errors\n";
	  return;
	}
	int trans[3];
	if(compare(endpoints[i].point[j],montiplet[k].orb[1][0].equiv[0].point[0],trans)){
	  for(int ii=0; ii<montiplet[k].orb.size(); ii++){
	    for(int jj=0; jj<montiplet[k].orb[ii].size(); jj++){
	      if(abs(montiplet[k].orb[ii][jj].eci) > 0.000000001){
		for(int kk=0; kk<montiplet[k].orb[ii][jj].equiv.size(); kk++){
		  for(int ll=0; ll<montiplet[k].orb[ii][jj].equiv[kk].point.size(); ll++){
		    mc_index tsite;
		    //first check whether the site is regular or activated before pushing back
		    if(basis[montiplet[k].orb[ii][jj].equiv[kk].point[ll].shift[3]].bit == 0){
		      for(int n=0; n<3; n++) tsite.shift[n]=montiplet[k].orb[ii][jj].equiv[kk].point[ll].shift[n]+trans[n];
		      tsite.shift[3]=montiplet[k].orb[ii][jj].equiv[kk].point[ll].shift[3];
		      if(new_mc_index(treach,tsite)) treach.push_back(tsite);
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    reach.push_back(treach);
  }
}

//************************************************************

void hop::print_hop_info(ostream &stream){
  stream << "hop for basis site " << b << "\n";
  stream << "The vacancy spin of the initial site is " << vac_spin_init << "\n";
  stream << "The initial site is: \n";
  initial.print(stream);
  stream << "\n";
  stream << "NUMBER of hops = " << endpoints.size() << "\n";
  stream << "The clusters corresponding to each hop are: \n";
  for(int i=0; i<endpoints.size(); i++){
    stream << "Cluster " << i << "\n";
    endpoints[i].print(stream);
  }
  stream << "\n";
  stream << "\n";
  for(int i=0; i<jump_vec.size(); i++){
    stream << "jump vector for hop " << i << "\n";
    jump_vec[i].print_cart(stream);
  }
  stream << "\n";
  stream << "\n";

  stream << "Shifts of the final states of the hops \n";
  for(int i=0; i<final.size(); i++){
    stream << "final state " << i << "  ";
    final[i].print(stream);
    stream << "\n";
  }
  stream << "\n";
  stream << "\n";

  stream << "Reach of the hop \n";
  for(int i=0; i< reach.size(); i++){
    stream << "Reach for hop " << i << "\n";
    for(int j=0; j< reach[i].size(); j++){
      reach[i][j].print(stream);
      stream << "\n";
    }
    stream << "\n";
  }
  stream << "\n";
  stream << "\n";

}


//************************************************************

void mc_index::print(ostream &stream){
  for(int i=0; i<4; i++) stream << shift[i] << "   ";

}




//************************************************************

void fluctuation::initialize(concentration conc){
  //dimensions the fluctuation object to be compatible with the concentration object conc
  f.clear();
  compon.clear();
  for(int i=0; i<conc.compon.size(); i++){
    vector<double> tf;
    vector< specie > tcompon;
    for(int j=0; j<conc.compon[i].size(); j++){
      tcompon.push_back(conc.compon[i][j]);
      for(int k=i; k<conc.compon.size(); k++){
        for(int l=0; l<conc.compon[k].size(); l++){
	  if(k!=i || l>=j)
            tf.push_back(0.0);
	}
      }
    }
    f.push_back(tf);
    compon.push_back(tcompon);
  }

}



//************************************************************

void fluctuation::set_zero(){
  for(int i=0; i<f.size(); i++){
    for(int j=0; j<f[i].size(); j++){
      f[i][j]=0.0;
    }
  }
}


//************************************************************

void fluctuation::evaluate(concentration conc){

  if(conc.compon.size() != f.size()){
    cout << "concentration and fluctuation variables are not compatible\n";
    cout << "no update of fluctuation\n";
    return;
  }
  for(int i=0; i<conc.compon.size(); i++){
    int m=0;
    for(int j=0; j<conc.compon[i].size(); j++){
      for(int k=i; k<conc.compon.size(); k++){
        for(int l=0; l<conc.compon[k].size(); l++){
	  if(k!=i || l>=j){
	    f[i][m]=conc.occup[i][j]*conc.occup[k][l];
	    m++;
	  }
	}
      }
    }
  }

}


//************************************************************

void fluctuation::evaluate(trajectory R){

  if(R.Rx.size() != f.size()){
    cout << "trajectory and fluctuation variables are not compatible\n";
    cout << "no update of fluctuation\n";
    return;
  }
  for(int i=0; i<R.Rx.size(); i++){
    int m=0;
    for(int j=0; j<R.Rx[i].size(); j++){
      for(int k=i; k<R.Rx.size(); k++){
        for(int l=0; l<R.Rx[k].size(); l++){
	  if(k!=i || l>=j){
	    f[i][m]=R.Rx[i][j]*R.Rx[k][l]+R.Ry[i][j]*R.Ry[k][l]+R.Rz[i][j]*R.Rz[k][l];
	    m++;
	  }
	}
      }
    }
  }

}



//************************************************************

void fluctuation::increment(fluctuation FF){

  if(f.size() != FF.f.size()){
    cout << "fluctuation variables are not compatible\n";
    cout << "cannot update \n";
    return;
  }

  for(int i=0; i<f.size(); i++){

    if(f[i].size() != FF.f[i].size()){
      cout << "fluctuation variables are not compatible\n";
      cout << "cannot update \n";
      return;
    }

    for(int j=0; j<f[i].size(); j++){
      f[i][j]=f[i][j]+FF.f[i][j];
    }
  }
}


//************************************************************

void fluctuation::decrement(fluctuation FF){

  if(f.size() != FF.f.size()){
    cout << "fluctuation variables are not compatible\n";
    cout << "cannot update \n";
    return;
  }

  for(int i=0; i<f.size(); i++){

    if(f[i].size() != FF.f[i].size()){
      cout << "fluctuation variables are not compatible\n";
      cout << "cannot update \n";
      return;
    }

    for(int j=0; j<f[i].size(); j++){
      f[i][j]=f[i][j]-FF.f[i][j];
    }
  }
}




//************************************************************

void fluctuation::normalize(double n){
  for(int i=0; i<f.size(); i++){
    for(int j=0; j<f[i].size(); j++){
      f[i][j]=f[i][j]/n;
    }
  }
}


//************************************************************

void fluctuation::normalize(double n, concentration conc){
  //for single component diffusion only normalizes the diagonal elements and sets the off-diagonal terms to zero
  if(compon.size() == 1 && compon[0].size() == 2){
    int m=0;
    for(int j=0; j< compon[0].size(); j++){
      for(int k=j; k< compon[0].size(); k++){
	if(j == k){
	  if(abs(conc.occup[0][j]) > tol) f[0][m]=f[0][m]/conc.occup[0][j];
	}
	else{
	  f[0][m]=0.0;
	}
	m++;
      }
    }
  }
  else{
    normalize(n);
  }

}



//************************************************************

void fluctuation::print(ostream &stream){
  if(compon.size() == 1 && compon[0].size() == 2){
    //only print the diagonal elements
    int m=0;
    for(int j=0; j<compon[0].size(); j++){
      for(int k=j; k<compon[0].size(); k++){
	if(j == k) stream << f[0][m] << "  ";
      }
    }
    m++;
  }

  else{
    for(int i=0; i<f.size(); i++){
      for(int j=0; j<f[i].size(); j++){
	stream << f[i][j] << "  ";
      }
    }
  }

}




//************************************************************

void fluctuation::print_elements(ostream &stream){

  if(compon.size() == 1 && compon[0].size() == 2){
    for(int j=0; j<compon[0].size(); j++){
      compon[0][j].print(stream);
      stream << "_";
      compon[0][j].print(stream);
      stream << "  ";
    }
  }
  else{
    for(int i=0; i<compon.size(); i++){
      for(int j=0; j<compon[i].size(); j++){
	for(int k=i; k<compon.size(); k++){
	  for(int l=0; l<compon[k].size(); l++){
	    if(k!=i || l>=j){
	      compon[i][j].print(stream);
	      stream << "_";
	      compon[k][l].print(stream);
	      stream << "  ";
	    }
	  }
	}
      }
    }
  }
}

//************************************************************

////////////////////////////////////////////////////////////////////////////////
//************************************************************
//added by Ben Swoboda
void get_clust_func(atompos atom1, atompos atom2, double &clust_func){
  int i,index;
  atom1.basis_vec.clear();

  index=atom2.bit;

  //determine which basis is to be used and store values in basis vector
  if(atom1.basis_flag == '1'){
    for(i=0; i<atom1.p_vec.size(); i++){
      atom1.basis_vec.push_back(atom1.p_vec[i]);
    }
    //        index++;
  }
  else{
    for(i=0; i<atom1.spin_vec.size(); i++){
      atom1.basis_vec.push_back(atom1.spin_vec[i]);
    }
  }

  clust_func=clust_func*atom1.basis_vec[index];

  basis_type=atom1.basis_flag;

  return;

}
//************************************************************
////////////////////////////////////////////////////////////////////////////////


//************************************************************

void calc_correlations(structure struc, multiplet super_basiplet, arrangement &conf){
  int nm,no,nc,np,na,i,j,k;
  double tcorr,clust_func;

  //push back the correlation for the empty cluster
  tcorr=1.0;
  conf.correlations.push_back(tcorr);

  for(nm=1; nm<super_basiplet.orb.size(); nm++){
    for(no=0; no<super_basiplet.orb[nm].size(); no++){
      tcorr=0;
      if(super_basiplet.orb[nm][no].equiv.size()==0) {
	cout << "something screwed up, no cluster in your orbit \n";
	exit(1);
      }
      for(nc=0; nc<super_basiplet.orb[nm][no].equiv.size(); nc++){
	clust_func=1;
	for(np=0; np<super_basiplet.orb[nm][no].equiv[nc].point.size(); np++){
	  for(na=0; na < struc.atom.size(); na++){
	    if(compare(struc.atom[na].fcoord,super_basiplet.orb[nm][no].equiv[nc].point[np].fcoord)){
	      //	      for(i=0; i<=super_basiplet.orb[nm][no].equiv[nc].point[np].bit; i++)
	      //		clust_func=clust_func*struc.atom[na].occ.spin;
	      //	      break;
	      ////////////////////////////////////////////////////////////////////////////////
	      //added by Ben Swoboda
	      //call get_clust_func => returns cluster function using desired basis
	      get_clust_func(struc.atom[na], super_basiplet.orb[nm][no].equiv[nc].point[np], clust_func);
	      //cout << "spin: " << struc.atom[na].occ.spin << "\tbit: " << super_basiplet.orb[nm][no].equiv[nc].point[np].bit <<
	      //"\tspecie: " << struc.atom[na].occ.name << "\n";
	      break;
	      ////////////////////////////////////////////////////////////////////////////////
	    }
	  }
	  if(na == struc.atom.size()){
	    cout << "have not mapped a cluster point on the crystal \n";
	    cout << "inside of calc_correlations \n";
	  }
	}
	tcorr=tcorr+clust_func;
      }
      tcorr=tcorr/super_basiplet.orb[nm][no].equiv.size();
      conf.correlations.push_back(tcorr);
    }
  }

  return;

}


//************************************************************
void get_super_basis_vec(structure &superstruc, vector < vector < vector < int > > > &super_basis_vec){
  int ns,i,j;

  /*Populate super_basis_vec, which contains the spin basis (including exponentiation or bit
    differentiation for ternary and higher order systems).  The super basis vector is ordered as follows:
    atom -> component specie -> basis components.  Currently supports spin and occupation bases, but
    will have to be edited to accomodate new bases.  Can be made more general if the data structures
    for storing occupation variable bases are generalized. */

  super_basis_vec.clear();

  //cout << "Printing super_basis_vec:  \n";
  for(i=0; i<superstruc.atom.size(); i++){
    vector < vector < int > > tcompon_vec;
    for(j=0; j<superstruc.atom[i].compon.size(); j++){
      superstruc.atom[i].occ=superstruc.atom[i].compon[j];
      vector<int> tbasis_vec;
      if(superstruc.atom[i].basis_flag=='1' && superstruc.atom[i].compon.size()>1){
	//cout << "\n Atom " << i << ": ";
	//cout << "Occupation basis ";
	get_basis_vectors(superstruc.atom[i]);
	for(int k=0; k<superstruc.atom[i].p_vec.size(); k++){
	  tbasis_vec.push_back(superstruc.atom[i].p_vec[k]);
	  //cout << "  " << tbasis_vec.back();
	}
      }
      else if(superstruc.atom[i].compon.size()>1){
	//cout << "\n Atom " << i << ": ";
	//cout << "Spin basis ";
	get_basis_vectors(superstruc.atom[i]);
	for(int k=0; k<superstruc.atom[i].spin_vec.size(); k++){
	  tbasis_vec.push_back(superstruc.atom[i].spin_vec[k]);
	  //cout << "  " << tbasis_vec.back() ;
	}
      }
      tcompon_vec.push_back(tbasis_vec);
    }
    super_basis_vec.push_back(tcompon_vec);
    superstruc.atom[i].occ=superstruc.atom[i].compon[0];
  }
  return;
}



//************************************************************

void get_corr_vector(structure &struc, multiplet &super_basiplet, vector< vector< vector< vector< int > > > > &corr_vec){
  int nm,no,nc,np,na,i,j,k;
  /*Go through super_basiplet and determine the basis functions for each correlation.  Store the site indeces and bit orderings
    for each local basis function of each correlation.  corr_vec is ordered as follows:
    basis function (as numbered in BCLUST) -> local equivalent basis functions -> atomic sites -> site number and bit/exponent value
  */

  corr_vec.clear();
  for(nm=1; nm<super_basiplet.orb.size(); nm++){
    for(no=0; no<super_basiplet.orb[nm].size(); no++){

      if(super_basiplet.orb[nm][no].equiv.size()==0) {
	cout << "something screwed up, no cluster in your orbit \n";
	exit(1);
      }
      vector < vector < vector < int > > > tcorr_vec;
      for(nc=0; nc<super_basiplet.orb[nm][no].equiv.size(); nc++){
	vector< vector< int > > func_vec;
	for(np=0; np<super_basiplet.orb[nm][no].equiv[nc].point.size(); np++){
	  for(na=0; na < struc.atom.size(); na++){
	    if(compare(struc.atom[na].fcoord,super_basiplet.orb[nm][no].equiv[nc].point[np].fcoord)){
	      vector< int > bit_vec;
	      bit_vec.push_back(na); //push back site index
	      bit_vec.push_back(super_basiplet.orb[nm][no].equiv[nc].point[np].bit); //push back bit value
	      func_vec.push_back(bit_vec);
	      break;
	    }
	  }
	  if(na == struc.atom.size()){
	    cout << "have not mapped a cluster point on the crystal \n";
	    cout << "inside of calc_correlations \n";
	  }
	}
	tcorr_vec.push_back(func_vec);
      }
      corr_vec.push_back(tcorr_vec);
    }
  }
  return;

}



//************************************************************

bool new_conf(arrangement &conf,superstructure &superstruc){

  for(int i=0; i<superstruc.conf.size(); i++){
    int j=0;
    while(j < superstruc.conf[i].correlations.size() &&
	  abs(conf.correlations[j]-superstruc.conf[i].correlations[j]) < tol){
      j++;
    }
    if(j == superstruc.conf[i].correlations.size()) return false;
  }
  return true;
}



//************************************************************

bool new_conf(arrangement &conf,vector<superstructure> &superstruc){

  for(int i=0; i<superstruc.size(); i++){
    if(!new_conf(conf,superstruc[i])) return false;
  }
  return true;
}




//************************************************************

void generate_ext_clust(structure struc, int min_num_compon, int max_num_points,
			vector<double> max_radius, multiplet &clustiplet){

  int i,j,k,m,n,np,nc;
  int dim[3];
  int num_basis=0;
  vector<atompos> basis;
  vector<atompos> gridstruc;



  //first get the basis sites

  {           // make the basis from which the sites for the local clusters are to be picked
    for(i=0; i<struc.atom.size(); i++)
      if(struc.atom[i].compon.size() >= min_num_compon){
	basis.push_back(struc.atom[i]);
	num_basis++;
      }
  }           // end of the basis generation



  //make the empty cluster
  //then starting from the empty cluster, start building multipoint clusters


  //make the empty cluster

  {          // beginning of the point cluster generation block
    vector<orbit> torbvec;
    cluster tclust;
    tclust.max_leng=0;
    tclust.min_leng=0;
    orbit torb;
    torb.equiv.push_back(tclust);
    torbvec.push_back(torb);
    clustiplet.orb.push_back(torbvec);
  }          // end of the empty cluster generation block


  //make a sphere with max_radius[2] and collect all crystal sites within that

  lat_dimension(struc.lat,max_radius[2],dim);



  for(i=-dim[0]; i<=dim[0]; i++){
    for(j=-dim[1]; j<=dim[1]; j++){
      for(k=-dim[2]; k<=dim[2]; k++){
        vec tlatt;

        tlatt.ccoord[0]=i*struc.lat[0][0]+j*struc.lat[1][0]+k*struc.lat[2][0];
        tlatt.ccoord[1]=i*struc.lat[0][1]+j*struc.lat[1][1]+k*struc.lat[2][1];
        tlatt.ccoord[2]=i*struc.lat[0][2]+j*struc.lat[1][2]+k*struc.lat[2][2];
        tlatt.cart_on=true;
        conv_AtoB(struc.CtoF,tlatt.ccoord,tlatt.fcoord);
        tlatt.frac_on=true;

        for(m=0; m<num_basis; m++){
          atompos tatom;
	  tatom=basis[m];
          for(int ii=0; ii<3; ii++){
            tatom.ccoord[ii]=basis[m].ccoord[ii]+tlatt.ccoord[ii];
            tatom.fcoord[ii]=basis[m].fcoord[ii]+tlatt.fcoord[ii];
          }

          //get distance to closest basis site in the unit cell at the origin

          double min_dist=1e20;
          for(n=0; n<num_basis; n++){
	    double temp[3];
	    double dist=0.0;
	    for(int ii=0; ii<3; ii++){
	      temp[ii]=tatom.ccoord[ii]-basis[n].ccoord[ii];
	      dist=dist+temp[ii]*temp[ii];
	    }
	    dist=sqrt(dist);
	    if(dist < min_dist)min_dist=dist;
          }
          if(min_dist < max_radius[2]) {
            gridstruc.push_back(tatom);
          }
        }
      }
    }
  }



  //for each cluster of the previous size, add points from gridstruc
  //   - see if the new cluster satisfies the size requirements
  //   - see if it is new
  //   - generate all its equivalents

  for(np=1; np<=max_num_points; np++){
    vector<orbit> torbvec;
    for(nc=0; nc<clustiplet.orb[np-1].size(); nc++){
      for(n=0; n<gridstruc.size(); n++){
        cluster tclust;
        atompos tatom;

        tatom=gridstruc[n];

        if(clustiplet.orb[np-1][nc].equiv.size() == 0){
          cout << "something screwed up \n";
          exit(1);
        }


        tclust=clustiplet.orb[np-1][nc].equiv[0];

        tclust.point.push_back(tatom);

        tclust.get_dimensions();

	if(tclust.point.size() == 1 && new_clust(tclust,torbvec)){
          orbit torb;
          torb.equiv.push_back(tclust);

          get_equiv(torb,struc.factor_group);

          torbvec.push_back(torb);
        }
	else{
	  if(tclust.max_leng < max_radius[np] && tclust.min_leng > tol && new_clust(tclust,torbvec)){
	    orbit torb;
	    torb.equiv.push_back(tclust);
	    get_equiv(torb,struc.factor_group);
	    torbvec.push_back(torb);
	  }
	}
      }
    }
    clustiplet.orb.push_back(torbvec);
    clustiplet.sort(np);
  }


}




//************************************************************


//************************************************************
void read_junk(istream &stream){  // added by jishnu //to skip the reading till the new line
  char j;
  do{
    stream.get(j);
  }while(!(stream.peek()=='\n'));
}
//************************************************************

void generate_loc_clust(structure struc, int min_num_compon, int max_num_points,
			vector<double> max_radius, multiplet &loc_clustiplet, cluster clust){

  int i,j,k,m,n,np,nc;
  int dim[3];
  int num_basis=0;
  vector<atompos> basis;
  vector<atompos> gridstruc;

  //first get the basis sites

  {           // make the basis from which the sites for the local clusters are to be picked
    for(i=0; i<struc.atom.size(); i++)
      if(struc.atom[i].compon.size() >= min_num_compon){
	basis.push_back(struc.atom[i]);
	num_basis++;
      }
  }           // end of the basis generation


  //now generate the local clusters emanating from input cluster

  //the first multiplet is simply the cluster itself

  {          // beginning of the point cluster generation block
    vector<orbit> torbvec;
    orbit torb;
    torb.equiv.push_back(clust);
    torbvec.push_back(torb);

    loc_clustiplet.orb.push_back(torbvec);
  }          // end of the empty cluster generation block




  //make a sphere with max_radius[2]+clust.max_leng and collect all crystal basis sites within that

  lat_dimension(struc.lat,max_radius[2]+clust.max_leng,dim);

  for(i=-dim[0]; i<=dim[0]; i++){
    for(j=-dim[1]; j<=dim[1]; j++){
      for(k=-dim[2]; k<=dim[2]; k++){
        vec tlatt;

        tlatt.ccoord[0]=i*struc.lat[0][0]+j*struc.lat[1][0]+k*struc.lat[2][0];
        tlatt.ccoord[1]=i*struc.lat[0][1]+j*struc.lat[1][1]+k*struc.lat[2][1];
        tlatt.ccoord[2]=i*struc.lat[0][2]+j*struc.lat[1][2]+k*struc.lat[2][2];
        tlatt.cart_on=true;
        conv_AtoB(struc.CtoF,tlatt.ccoord,tlatt.fcoord);
        tlatt.frac_on=true;

        for(m=0; m<num_basis; m++){
          atompos tatom;
	  tatom=basis[m];
          for(int ii=0; ii<3; ii++){
            tatom.ccoord[ii]=basis[m].ccoord[ii]+tlatt.ccoord[ii];
            tatom.fcoord[ii]=basis[m].fcoord[ii]+tlatt.fcoord[ii];
          }

          //get distance to the site of interest atom_num

	  double min_dist=1.0e20;
	  for(n=0; n<clust.point.size(); n++){
	    double temp[3];
	    double dist=0.0;
	    for(int ii=0; ii<3; ii++){
	      temp[ii]=tatom.ccoord[ii]-clust.point[n].ccoord[ii];
	      dist=dist+temp[ii]*temp[ii];
	    }
	    dist=sqrt(dist);
	    if(dist < min_dist)min_dist=dist;
	  }
	  if(min_dist < max_radius[2]) {
	    gridstruc.push_back(tatom);
	  }
	}
      }
    }
  }

  //for each cluster of the previous size, add points from gridstruc
  //   - see if the new cluster satisfies the size requirements
  //   - see if it is new
  //   - generate all its equivalents

  for(np=1; np<=max_num_points-1; np++){
    vector<orbit> torbvec;

    for(nc=0; nc<loc_clustiplet.orb[np-1].size(); nc++){
      for(n=0; n<gridstruc.size(); n++){
        cluster tclust;
        atompos tatom;

        tatom=gridstruc[n];

        if(loc_clustiplet.orb[np-1][nc].equiv.size() == 0){
          cout << "something screwed up \n";
          exit(1);
        }

        tclust=loc_clustiplet.orb[np-1][nc].equiv[0];
        tclust.point.push_back(tatom);

        tclust.get_dimensions();

        if(tclust.max_leng < max_radius[np+1] && tclust.min_leng > tol && new_loc_clust(tclust,torbvec)){
          orbit torb;
          torb.equiv.push_back(tclust);
          get_loc_equiv(torb,clust.clust_group);
          torbvec.push_back(torb);
        }
      }
    }
    loc_clustiplet.orb.push_back(torbvec);
    loc_clustiplet.sort(np);
  }

  cout << "\n";
  cout << "LOCAL CLUSTER \n";
  cout << "\n";

  for(np=1; np<=max_num_points-1; np++)
    loc_clustiplet.print(cout);

}


//************************************************************

void calc_clust_symmetry(structure struc, cluster &clust){
  int pg,na,i,j,k,n,m,num_suc_maps;
  atompos hatom;
  double shift[3],temp[3];
  sym_op tclust_group;
  cluster tclust;


  //all symmetry operations are done within the fractional coordinate system
  //since translations back into the unit cell are straightforward


  //apply a point group operation to the cluster
  //then see if a translation (shift) maps the transformed cluster onto the original cluster
  //if so test whether this point group operation and translation maps the crystal onto itself


  for(pg=0; pg<struc.point_group.size(); pg++){

    tclust=clust.apply_sym(struc.point_group[pg]);

    for(i=0; i<clust.point.size(); i++){
      if(compare(clust.point[0].compon, tclust.point[i].compon)){

	for(j=0; j<3; j++) shift[j]=clust.point[0].fcoord[j]-tclust.point[i].fcoord[j];

	//check whether the cluster maps onto itself

	num_suc_maps=0;
	for(n=0; n<clust.point.size(); n++){
	  for(m=0; m<clust.point.size(); m++){
	    if(compare(clust.point[n].compon, clust.point[m].compon)){
	      for(j=0; j<3; j++) temp[j]=clust.point[n].fcoord[j]-tclust.point[m].fcoord[j]-shift[j];

	      k=0;
	      for(j=0; j<3; j++)
		if(abs(temp[j]) < 0.00005 ) k++;
	      if(k==3)num_suc_maps++;
	    }
	  }
	}

	if(num_suc_maps == clust.point.size()){
	  //the cluster after transformation and translation maps onto itself
	  //now check whether the rest of the crystal maps onto itself

	  //apply the point group to the crystal

	  vector<atompos> tatom;
	  for(na=0; na<struc.atom.size(); na++){
	    hatom=struc.atom[na].apply_sym(struc.point_group[pg]);
	    tatom.push_back(hatom);
	  }
	  //check whether after translating with shift it maps onto itself
	  num_suc_maps=0;
	  for(n=0; n<struc.atom.size(); n++){
	    for(m=0; m<struc.atom.size(); m++){
	      if(compare(struc.atom[n].compon,struc.atom[m].compon)){
		for(j=0; j<3; j++) temp[j]=struc.atom[n].fcoord[j]-tatom[m].fcoord[j]-shift[j];
		within(temp);

		k=0;
		for(j=0; j<3; j++)
		  if(abs(temp[j]) < 0.00005 ) k++;
		if(k==3)num_suc_maps++;
	      }
	    }
	  }

	  if(num_suc_maps == struc.atom.size()){

	    //check whether the symmetry operation already exists in the factorgroup array

	    int ll=0;
	    for(int ii=0; ii<clust.clust_group.size(); ii++)
	      if(compare(struc.point_group[pg].fsym_mat,clust.clust_group[ii].fsym_mat)
		 && compare(shift,clust.clust_group[ii].ftau) )break;
	      else ll++;

	    // if the symmetry operation is new, add it to the clust_group vector
	    // and update all info about the sym_op object

	    if(clust.clust_group.size() == 0 || ll == clust.clust_group.size()){
	      tclust_group.frac_on=false;
	      tclust_group.cart_on=false;
	      for(int jj=0; jj<3; jj++){
		tclust_group.ftau[jj]=shift[jj];
		for(int kk=0; kk<3; kk++){
		  tclust_group.fsym_mat[jj][kk]=struc.point_group[pg].fsym_mat[jj][kk];
		  tclust_group.lat[jj][kk]=struc.lat[jj][kk];
		}
	      }
	      tclust_group.frac_on=true;
	      tclust_group.update();
	      clust.clust_group.push_back(tclust_group);
	    }
	  }
	  tatom.clear();
	}
      }
    }
  }
  return;
}

//************************************************************

void generate_ext_basis_environ(structure struc, multiplet clustiplet, multiplet &basiplet){  // jishnu
  int np,no,i,j,k;

  //go through the clustiplet
  //for each orbit within the clustiplet, take the first of the equivalent clusters,
  //  enumerate each exponent sequence for that cluster
  //     first check if the exponent sequence on the cluster is new compared with already considered sequences
  //     for each new exponent sequence, generate all equivalent basis clusters by doing:
  //                 -for each factor_group
  //                    apply factor_group to the cluster
  //                    determine the cluster group symmetry
  //                    for each clust_group
  //                       apply clust_group to the cluster
  //                       determine if a new cluster


  if(clustiplet.orb.size() > 0){
    basiplet.orb.push_back(clustiplet.orb[0]);
  }

  for(np=1; np<clustiplet.orb.size(); np++){
    vector<orbit> torbvec;
    for(no=0; no<clustiplet.orb[np].size(); no++){
      cluster tclust;
      tclust=clustiplet.orb[np][no].equiv[0];

      //enumerate each exponent sequence for this cluster
      int last=0;

      while(last == 0){
	tclust.point[0].bit++;
	for(i=0; i<(tclust.point.size()-1); i++){
	  if(tclust.point[i].bit !=0 && tclust.point[i].bit%tclust.point[i].compon.size() == 0){   // changed by jishnu // got rid of '-1' to get all possible bits (considering Vacancies as one component)
	    tclust.point[i+1].bit++;
	    tclust.point[i].bit=0;
	  }
	}
	if(tclust.point[tclust.point.size()-1].bit !=0 &&
	   tclust.point[tclust.point.size()-1].bit%tclust.point[tclust.point.size()-1].compon.size() == 0){ // changed by jishnu // got rid of '-1' to get all possible bits (considering Vacancies as one component)
	  last=last+1;
	  tclust.point[tclust.point.size()-1].bit=0;
	}


	// check if this cluster already exists
	if(new_clust(tclust, torbvec)){

	  //if not apply factor group

	  orbit torb;
	  for(int fg=0; fg<struc.factor_group.size(); fg++){
	    cluster tclust1;
	    tclust1=tclust.apply_sym(struc.factor_group[fg]);
	    within(tclust1);
	    tclust1.get_cart(struc.FtoC);

	    //determine cluster symmetry

	    calc_clust_symmetry(struc,tclust1);

	    //apply cluster symmetry and check if already part of current orbit

	    for(int cg=0; cg<tclust1.clust_group.size(); cg++){
	      cluster tclust2;
	      tclust2=tclust1.apply_sym(tclust1.clust_group[cg]);
	      tclust2.get_cart(struc.FtoC);

	      if(new_clust(tclust2,torb)){
		torb.equiv.push_back(tclust2);
	      }
	    }
	  }
	  if(torb.equiv.size() !=0){
	    torbvec.push_back(torb);
	  }
	}
      }
    }
    basiplet.orb.push_back(torbvec);
  }
  return;
}



////////////////////////////////////////////////////////////////////////////////


//************************************************************

void generate_ext_basis(structure struc, multiplet clustiplet, multiplet &basiplet){
  int np,no,i,j,k;

  //go through the clustiplet
  //for each orbit within the clustiplet, take the first of the equivalent clusters,
  //  enumerate each exponent sequence for that cluster
  //     first check if the exponent sequence on the cluster is new compared with already considered sequences
  //     for each new exponent sequence, generate all equivalent basis clusters by doing:
  //                 -for each factor_group
  //                    apply factor_group to the cluster
  //                    determine the cluster group symmetry
  //                    for each clust_group
  //                       apply clust_group to the cluster
  //                       determine if a new cluster


  if(clustiplet.orb.size() > 0){
    basiplet.orb.push_back(clustiplet.orb[0]);
  }

  for(np=1; np<clustiplet.orb.size(); np++){
    vector<orbit> torbvec;
    for(no=0; no<clustiplet.orb[np].size(); no++){
      cluster tclust;
      tclust=clustiplet.orb[np][no].equiv[0];

      //enumerate each exponent sequence for this cluster
      int last=0;

      while(last == 0){
	tclust.point[0].bit++;
	for(i=0; i<(tclust.point.size()-1); i++){
	  if(tclust.point[i].bit !=0 && tclust.point[i].bit%(tclust.point[i].compon.size()-1) == 0){
	    tclust.point[i+1].bit++;
	    tclust.point[i].bit=0;
	  }
	}
	if(tclust.point[tclust.point.size()-1].bit !=0 &&
	   tclust.point[tclust.point.size()-1].bit%(tclust.point[tclust.point.size()-1].compon.size()-1) == 0){
	  last=last+1;
	  tclust.point[tclust.point.size()-1].bit=0;
	}


	// check if this cluster already exists
	if(new_clust(tclust, torbvec)){

	  //if not apply factor group

	  orbit torb;
	  for(int fg=0; fg<struc.factor_group.size(); fg++){
	    cluster tclust1;
	    tclust1=tclust.apply_sym(struc.factor_group[fg]);
	    within(tclust1);
	    tclust1.get_cart(struc.FtoC);

	    //determine cluster symmetry

	    calc_clust_symmetry(struc,tclust1);

	    //apply cluster symmetry and check if already part of current orbit

	    for(int cg=0; cg<tclust1.clust_group.size(); cg++){
	      cluster tclust2;
	      tclust2=tclust1.apply_sym(tclust1.clust_group[cg]);
	      tclust2.get_cart(struc.FtoC);

	      if(new_clust(tclust2,torb)){
		torb.equiv.push_back(tclust2);
	      }
	    }
	  }
	  if(torb.equiv.size() !=0){
	    torbvec.push_back(torb);
	  }
	}
      }
    }
    basiplet.orb.push_back(torbvec);
  }
  return;
}



////////////////////////////////////////////////////////////////////////////////
//added by anton - filters a multiplet for clusters containing just one activated site (with occupation basis = 1)

//************************************************************
void filter_activated_clust(multiplet clustiplet, multiplet &activatedclustiplet){


  //clear activatedclustiplet
  activatedclustiplet.orb.clear();
  activatedclustiplet.size.clear();
  activatedclustiplet.order.clear();
  activatedclustiplet.index.clear();
  activatedclustiplet.subcluster.clear();

  //copy the empty cluster into activatedclustiplet

  if(clustiplet.orb.size() >= 1){
    activatedclustiplet.orb.push_back(clustiplet.orb[0]);
  }


  for(int i=1; i<clustiplet.orb.size(); i++){
    vector<orbit> torb;
    for(int j=0; j<clustiplet.orb[i].size(); j++){
      int num_activated=0;
      for(int k=0; k<clustiplet.orb[i][j].equiv[0].point.size(); k++){
	if(clustiplet.orb[i][j].equiv[0].point[k].basis_flag == '1') {
	  num_activated++;
	}
      }
      if(num_activated == 1){
	torb.push_back(clustiplet.orb[i][j]);
      }
    }
    activatedclustiplet.orb.push_back(torb);
  }


}

////////////////////////////////////////////////////////////////////////////////

//added by anton
//************************************************************

void merge_multiplets(multiplet clustiplet1, multiplet clustiplet2, multiplet &clustiplet3){

  //merge clustiplet1 with clustiplet2 and put it in clustiplet3


  for(int np=0; np<clustiplet1.orb.size(); np++){
    clustiplet3.orb.push_back(clustiplet1.orb[np]);
  }


  for(int np=1; np<clustiplet2.orb.size(); np++){
    if(np> clustiplet3.orb.size()) clustiplet3.orb.push_back(clustiplet2.orb[np]);
    else{
      //add only orbits from clustiplet2.orb[np] that are new
      for(int i=0; i<clustiplet2.orb[np].size(); i++){
	bool isnew = true;
	for(int j=0; j<clustiplet3.orb[np].size(); j++){
	  if(compare(clustiplet2.orb[np][i],clustiplet3.orb[np][j])){
	    isnew = false;
	    break;
	  }
	}
	if(isnew) clustiplet3.orb[np].push_back(clustiplet2.orb[np][i]);
      }
    }
  }

  for(int np=1; np < clustiplet3.orb.size(); np++){
    clustiplet3.sort(np);
  }

}




//************************************************************

void generate_ext_monteclust(vector<atompos> basis, multiplet basiplet, vector<multiplet> &montiplet){


  //takes the basiplet and for each basis site in prim and enumerates all clusters that go through that basis site
  //then it determines the shift indices for each point of these clusters

  //for each basis site (make sure the basis site is within the primitive unit cell)
  //go through the basiplet and for each cluster within an orbit
  //        translate the cluster such that each point has been within the primitive unit cell
  //        if the resulting cluster has a point that coincides with the basis site,
  //        add it to the orbit for that cluster type in montiplet
  //        also determine the shift indices for this cluster

  for(int na=0; na<basis.size(); na++){   //select an atompos from prim, LOOP in atom

    multiplet tmontiplet;
    int clust_count=0;           //Edited by John -> Track mapping of clusters to basis sites
    //make sure the basis atom is within the primitive unit cell
    within(basis[na]);

    //first make an empty cluster and add it to tmontiplet
    {
      cluster tclust;
      orbit torb;  // has a non-equivalent cluster, a colection of equivalent clusters, and ECI value
      torb.eci=basiplet.orb[0][0].eci;  // ECI of empty cluster
      torb.equiv.push_back(tclust);     // push empty cluster
      vector<orbit>torbvec;
      torbvec.push_back(torb);
      tmontiplet.orb.push_back(torbvec);

      vector<int> tind_vec;               //Edited code
      tind_vec.push_back(clust_count);    // Edited code
      tmontiplet.index.push_back(tind_vec);  //Edited code
      clust_count++; // Edited code
    }

    //go through each cluster of basiplet
    for(int np=1; np<basiplet.orb.size(); np++){    // select one row of orb table, from row 1, LOOP in orb
      vector<orbit> torbvec;
      vector<int> tind_vec;  // Edited code

      for(int no=0; no<basiplet.orb[np].size(); no++){  // select each clumn in a given row of orb table, LOOP orbit
	orbit torb;
	torb.eci=basiplet.orb[np][no].eci;
	bool found=false;
	for(int neq=0; neq<basiplet.orb[np][no].equiv.size(); neq++){  // LOOP equiv.size()
	  cluster tclust=basiplet.orb[np][no].equiv[neq];   // select an equivalent cluster

	  //for each point of the cluster translate the cluster so that point lies
	  //within the primitive unit cell

	  for(int n=0; n<tclust.point.size(); n++){ //LOOP n<np, np is the row index of orb table, also is the size of cluster, 1 means point
	    within(tclust,n);      //translate nth point of a cluster into prim cell, other points translate accordingly
	    //check whether the basis site basis[na] belongs to this cluster
	    if(compare(tclust.point[n].fcoord, basis[na].fcoord)){  //if the selected point belongs to a given basis point, then push this cluster
	      //add the cluster to the orbit
	      torb.equiv.push_back(tclust);  // if the given basis site basis[na] belongs this cluster, then push this cluster
	      found=true;
	    }
	  }  // LOOP n<np


	}    // LOOP equiv.size()
	if(found){
	  torbvec.push_back(torb);
	  tind_vec.push_back(clust_count); //Edited Code -- map index of cluster in basiplet to index of cluster in montiplet
	}
	clust_count++;
      }  // LOOP orbit
      tmontiplet.orb.push_back(torbvec);
      tmontiplet.index.push_back(tind_vec);  // Edited code
    }   //LOOP in orb
    montiplet.push_back(tmontiplet);
  } // LOOP atom
  //after the above part, each basis atompos has a orb table.



  //work out the shift tables in each cluster object
  //these tell us which basis the point of the cluster belongs to and the coordinates of the unit cell

  for(int nm=0; nm < montiplet.size(); nm++){  // LOOP start
    for(int np=0; np < montiplet[nm].orb.size(); np++){
      for(int no=0; no < montiplet[nm].orb[np].size(); no++){
        for(int ne=0; ne < montiplet[nm].orb[np][no].equiv.size(); ne++){
          for(int n=0; n < montiplet[nm].orb[np][no].equiv[ne].point.size(); n++){

            get_shift(montiplet[nm].orb[np][no].equiv[ne].point[n], basis);

          }
        }
      }
    }
  }  // LOOP end
}






//************************************************************
//Useful functions
//************************************************************

void double_to_string(double n, string &a, int dec_places){
  //Only works for base 10 numbers
  double nn;
  int i;
  if(n<0)
    a.push_back('-');
  n=abs(n);
  nn=floor(n);
  i=int(nn);
  int_to_string(i, a, 10);
  if(dec_places>0)
    a.push_back('.');
  while(dec_places>0){
    n=10*(n-nn);
    nn=floor(n);
    i=int(nn);
    int_to_string(i, a, 10);
    dec_places--;
  }
  return;
}

//************************************************************


void int_to_string(int i, string &a, int base){
  int ii=i;
  string aa;

  if(ii==0){
    a.push_back(ii+48);
    return;
  }

  if(ii<0)a.push_back('-');
  ii=abs(ii);

  int remain=ii%base;

  while(ii > 0){
    aa.push_back(ii%base+48);
    ii=(ii-remain)/base;
    remain=ii%base;
  }
  for(ii=aa.size()-1; ii>=0; ii--){
    a.push_back(aa[ii]);
  }
  return;
}


//************************************************************

double determinant(double mat[3][3]){
  return mat[0][0]*(mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1])-
    mat[0][1]*(mat[1][0]*mat[2][2]-mat[1][2]*mat[2][0])+
    mat[0][2]*(mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0]);
}

//************************************************************

void inverse(double mat[3][3], double invmat[3][3]){
  double det=determinant(mat);
  invmat[0][0]=(+mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1])/det;
  invmat[0][1]=(-mat[0][1]*mat[2][2]+mat[0][2]*mat[2][1])/det;
  invmat[0][2]=(+mat[0][1]*mat[1][2]-mat[0][2]*mat[1][1])/det;
  invmat[1][0]=(-mat[1][0]*mat[2][2]+mat[1][2]*mat[2][0])/det;
  invmat[1][1]=(+mat[0][0]*mat[2][2]-mat[0][2]*mat[2][0])/det;
  invmat[1][2]=(-mat[0][0]*mat[1][2]+mat[0][2]*mat[1][0])/det;
  invmat[2][0]=(+mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0])/det;
  invmat[2][1]=(-mat[0][0]*mat[2][1]+mat[0][1]*mat[2][0])/det;
  invmat[2][2]=(+mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0])/det;
  return;
}

//************************************************************
//mat3=mat1*mat2
void matrix_mult(double mat1[3][3], double mat2[3][3], double mat3[3][3]){
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      mat3[i][j]=0.0;
      for(int k=0; k<3; k++){
	mat3[i][j]=mat3[i][j]+mat1[i][k]*mat2[k][j];
      }
    }
  }
}


//************************************************************
//Added by John
//Given vector vec1, find a perpendicular vector vec2
void get_perp(double vec1[3], double vec2[3]){

  for(int i=0; i<3; i++){
    vec2[i]=0.0;
  }
  for(int i=0; i<3; i++){
    if(abs(vec1[i])<tol){
      vec2[i]=1.0;
      return;
    }
  }

  vec2[0]=vec2[1]=1.0;
  vec2[2]=-(vec1[0]+vec1[1])/vec1[2];
  normalize(vec2,1.0);
  return;
}
//\End Addition

//************************************************************
//Added by John
//Given vectors vec1 and vec2, find perpendicular vector vec3
void get_perp(double vec1[3], double vec2[3], double vec3[3]){
  for(int i=0; i<3; i++)
    vec3[i]=vec1[(i+1)%3]*vec2[(i+2)%3]-vec1[(i+2)%3]*vec2[(i+1)%3];
  if(normalize(vec3, 1.0))
    return;
  else get_perp(vec1, vec3);
}
//\End Addition

//************************************************************
//Added by John
bool normalize(double vec1[3], double length){
  double tmag=0.0;
  for(int i=0; i<3; i++){
    tmag+=vec1[i]*vec1[i];
  }
  tmag=sqrt(tmag);
  if(tmag>tol){
    for(int i=0; i<3; i++){
      if(abs(vec1[i])>tol)
	vec1[i]*=length/tmag;
      else vec1[i]=0.0;
    }
    return true;
  }
  else return false;
}
//\End addition
//************************************************************

void coord_trans_mat(double lat[3][3], double FtoC[3][3], double CtoF[3][3]){
  int i,j;

  for(i=0; i<3; i++)
    for(j=0; j<3; j++)
      FtoC[i][j]=lat[j][i];

  inverse(FtoC,CtoF);
}


//************************************************************

bool compare(double mat1[3][3], double mat2[3][3]){
  int i,j,k;

  k=0;
  for(i=0; i<3; i++)
    for(j=0; j<3; j++)
      if(abs(mat1[i][j]-mat2[i][j]) < tol) k++;

  if(k == 9) return true;
  else return false;
}


//************************************************************

bool compare(double vec1[3], double vec2[3]){
  int i,k;

  k=0;
  for(i=0; i<3; i++)
    if(abs(vec1[i]-vec2[i]) < tol) k++;

  if(k == 3) return true;
  else return false;
}


//************************************************************

bool compare(double vec1[3], double vec2[3], int trans[3]){

  double ftrans[3];
  for(int i=0; i<3; i++){
    ftrans[i]=vec1[i]-vec2[i];
    trans[i]=0;
    //check whether all elements of ftrans[3] are within a tolerance of an integer
    while(abs(ftrans[i]) > tol && ftrans[i] > 0){
      ftrans[i]=ftrans[i]-1.0;
      trans[i]=trans[i]+1;
    }
    while(abs(ftrans[i]) > tol && ftrans[i] < 0){
      ftrans[i]=ftrans[i]+1.0;
      trans[i]=trans[i]-1;
    }
    if(abs(ftrans[i] > tol)) return false;
  }
  return true;
}


//************************************************************

bool compare(char name1[2], char name2[2]){
  for(int i=0; i<2; i++)
    if(name1[i] != name2[i]) return false;
  return true;
}


//************************************************************

bool compare(specie compon1, specie compon2){   // changed by jishnu
  int i,k;

  //k=0;
  //for(i=0; i<2; i++)
  //  if(compon1.name[i] == compon2.name[i]) k++;
  //if(k == 2) return true;
  if(compon1.name.compare(compon2.name) == 0) return true;
  else return false;
}


//************************************************************

bool compare(vector<specie> compon1, vector<specie> compon2){
  int i,j,k;
  int num_suc_maps,ll;

  if(compon1.size() != compon2.size()) return false;

  num_suc_maps=0;
  for(i=0; i<compon1.size(); i++){
    for(j=0; j<compon2.size(); j++){
      if(compare(compon1[i],compon2[j])) num_suc_maps++;
    }
  }
  if(num_suc_maps == compon1.size()) return true;
  else return false;
}


//************************************************************

bool compare(atompos &atom1, atompos &atom2){
  int i,j,k,l;

  k=0;
  l=0;
  for(i=0; i<3; i++)
    if(abs(atom1.fcoord[i]-atom2.fcoord[i]) < tol) k++;
    else return false;

  if(k == 3 && compare(atom1.compon, atom2.compon) &&
     atom1.bit == atom2.bit) return true;
  else return false;

}

//************************************************************

bool compare_just_coordinates(atompos &atom1, atompos &atom2){
  int i,j,k,l;

  k=0;
  l=0;
  for(i=0; i<3; i++)
    if(abs(atom1.fcoord[i]-atom2.fcoord[i]) < tol) k++;
    else return false;

  //if(k == 3 && compare(atom1.compon, atom2.compon) ) return true;
  if(k == 3) return true;
  else return false;

}



//************************************************************

bool compare(atompos atom1, atompos atom2, int trans[3]){

  double ftrans[3];
  for(int i=0; i<3; i++){
    ftrans[i]=atom1.fcoord[i]-atom2.fcoord[i];
    trans[i]=0;
    //check whether all elements of ftrans[3] are within a tolerance of an integer
    while(abs(ftrans[i]) > tol && ftrans[i] > 0){
      ftrans[i]=ftrans[i]-1.0;
      trans[i]=trans[i]+1;
    }
    while(abs(ftrans[i]) > tol && ftrans[i] < 0){
      ftrans[i]=ftrans[i]+1.0;
      trans[i]=trans[i]-1;
    }
    if(abs(ftrans[i] > tol)) return false;
  }
  return true;
}
//************************************************************

bool compare_just_coordinates(cluster &clust1, cluster &clust2){

  ////////////////////////////////////////////////////////////////////////////////
  //added by anton
  if(clust1.point.size() != clust2.point.size()) return false;

  int k=0;
  for(int np1=0; np1<clust1.point.size(); np1++){
    for(int np2=0; np2<clust2.point.size(); np2++){
      if(compare_just_coordinates(clust1.point[np1],clust2.point[np2])) k++;
    }
  }

  if(k == clust1.point.size()) return true;
  else return false;
}



//************************************************************

//************************************************************

bool compare(cluster &clust1, cluster &clust2){

  ////////////////////////////////////////////////////////////////////////////////
  //added by anton
  if(clust1.point.size() != clust2.point.size()) return false;

  int k=0;
  for(int np1=0; np1<clust1.point.size(); np1++){
    for(int np2=0; np2<clust2.point.size(); np2++){
      if(compare(clust1.point[np1],clust2.point[np2])) k++;
    }
  }

  if(k == clust1.point.size()) return true;
  else return false;
}



//************************************************************
////////////////////////////////////////////////////////////////////////////////
//added by anton
bool compare(orbit orb1, orbit orb2){

  if(orb1.equiv.size() != orb2.equiv.size()) return false;

  int k=0;
  for(int ne1=0; ne1<orb1.equiv.size(); ne1++){
    for(int ne2=0; ne2<orb2.equiv.size(); ne2++){
      if(compare(orb1.equiv[ne1],orb2.equiv[ne2])) k++;
    }
  }

  if(k == orb1.equiv.size()) return true;
  else return false;
}




////////////////////////////////////////////////////////////////////////////////




//************************************************************

bool compare(vector<double> vec1, vector<double> vec2){
  if(vec1.size() != vec2.size()) return false;
  for(int i=0; i<vec1.size(); i++)
    if(abs(vec1[i]-vec2[i]) > tol) return false;

  return true;

}


//************************************************************

bool compare(concentration conc1, concentration conc2){
  if(conc1.compon.size() != conc2.compon.size()) return false;
  for(int i=0; i<conc1.compon.size(); i++){
    if(!compare(conc1.compon[i],conc2.compon[i])) return false;
    if(!compare(conc1.occup[i],conc2.occup[i]))return false;
  }

  return true;

}


//************************************************************

bool compare(mc_index m1, mc_index m2){
  for(int i=0; i<4; i++)
    if(m1.shift[i] != m2.shift[i]) return false;

  return true;

}





//************************************************************

bool new_mc_index(vector<mc_index> v1, mc_index m2){
  for(int i=0; i<v1.size(); i++)
    if(compare(v1[i],m2)) return false;

  return true;
}


//************************************************************

bool is_integer(double vec[3]){
  int j,k;
  k=0;
  for(j=0; j<3; j++)
    if(abs(vec[j]-ceil(vec[j])) < tol || abs(vec[j]-floor(vec[j])) < tol) k++;

  if(k == 3) return true;
  else return false;
}


//************************************************************

bool is_integer(double mat[3][3]){
  int i,j,k;
  k=0;
  for(i=0; i<3; i++)
    for(j=0; j<3; j++)
      if(abs(mat[i][j]-ceil(mat[i][j])) < tol || abs(mat[i][j]-floor(mat[i][j])) < tol) k++;

  if(k == 9) return true;
  else return false;
}




//************************************************************

void within(double fcoord[3]){
  int i;
  for(i=0; i<3; i++){
    while(fcoord[i] < 0.0)fcoord[i]=fcoord[i]+1.0;
    while(fcoord[i] >0.99999)fcoord[i]=fcoord[i]-1.0;
  }
  return;
}


//************************************************************

void within(atompos &atom){
  int i;
  for(i=0; i<3; i++){
    while(atom.fcoord[i] < 0.0)atom.fcoord[i]=atom.fcoord[i]+1.0;
    while(atom.fcoord[i] >0.99999)atom.fcoord[i]=atom.fcoord[i]-1.0;
  }
  return;
}

//************************************************************
//added by Ben Swoboda
// used to translate all PRIM coordinates to unit cell

void within(structure &struc){
  int i;
  for(i=0; i<struc.atom.size(); i++){
    within(struc.atom[i]);
  }
  return;
}

//************************************************************
// translates a cluster so that its first point is within the unit cell

void within(cluster &clust){
  int i,np;
  for(i=0; i<3; i++){
    while(clust.point[0].fcoord[i] < 0.0){
      clust.point[0].fcoord[i]=clust.point[0].fcoord[i]+1.0;
      for(np=1; np<clust.point.size(); np++)
	clust.point[np].fcoord[i]=clust.point[np].fcoord[i]+1.0;
    }
    while(clust.point[0].fcoord[i] >0.99999){
      clust.point[0].fcoord[i]=clust.point[0].fcoord[i]-1.0;
      for(np=1; np<clust.point.size(); np++)
	clust.point[np].fcoord[i]=clust.point[np].fcoord[i]-1.0;
    }
  }
  return;
}


//************************************************************
// translates a cluster so that its nth  point is within the unit cell

void within(cluster &clust, int n){
  int i,np;
  for(i=0; i<3; i++){
    while(clust.point[n].fcoord[i] < 0.0){
      for(np=0; np<clust.point.size(); np++)
	clust.point[np].fcoord[i]=clust.point[np].fcoord[i]+1.0;
    }
    while(clust.point[n].fcoord[i] > 0.99999){
      for(np=0; np<clust.point.size(); np++)
	clust.point[np].fcoord[i]=clust.point[np].fcoord[i]-1.0;
    }
  }
  return;
}




//************************************************************
/*
  This function, given a matrix with as rows the cartesian coordinates
  of the unit cell vectors defining a lattice, returns the lengths of the
  unit cell vectors (in latparam) and the angles between the vectors (latparam)
*/
//************************************************************

void latticeparam(double lat[3][3], double latparam[3], double latangle[3])
{
  int i,j;
  double temp;

  //calculate the a,b,c lattice parameters = length of the unit cell vectors

  for(i=0; i<3; i++){
    latparam[i]=0.0;
    for(j=0; j<3; j++)latparam[i]=latparam[i]+lat[i][j]*lat[i][j];
    latparam[i]=sqrt(latparam[i]);
  }

  //calculate the angles between the unit cell vectors

  for(i=0; i<3; i++){
    latangle[i]=0.0;
    for(j=0; j<3; j++) latangle[i]=latangle[i]+lat[(i+1)%3][j]*lat[(i+2)%3][j];
    temp=latangle[i]/(latparam[(i+1)%3]*latparam[(i+2)%3]);

    //make sure numerical errors don't place the arguments outside of the [-1,1] interval

    if((temp-1.0) > 0.0)temp=1.0;
    if((temp+1.0) < 0.0)temp=-1.0;
    latangle[i]=(180.0/3.141592654)*acos(temp);
  }

  return;
}


//************************************************************
/*
  This function, given a matrix with as rows the cartesian coordinates
  of the unit cell vectors defining a lattice, returns the lengths of the
  unit cell vectors (in latparam) and the angles between the vectors (latparam)
  It also determines which vector is largest, smallest and in between in length.
*/
//************************************************************

void latticeparam(double lat[3][3], double latparam[3], double latangle[3], int permut[3])
{
  int i,j;
  double temp;

  //calculate the a,b,c lattice parameters = length of the unit cell vectors

  for(i=0; i<3; i++){
    latparam[i]=0.0;
    for(j=0; j<3; j++)latparam[i]=latparam[i]+lat[i][j]*lat[i][j];
    latparam[i]=sqrt(latparam[i]);
  }

  //calculate the angles between the unit cell vectors

  for(i=0; i<3; i++){
    latangle[i]=0.0;
    for(j=0; j<3; j++) latangle[i]=latangle[i]+lat[(i+1)%3][j]*lat[(i+2)%3][j];
    temp=latangle[i]/(latparam[(i+1)%3]*latparam[(i+2)%3]);

    //make sure numerical errors don't place the arguments outside of the [-1,1] interval

    if((temp-1.0) > 0.0)temp=1.0;
    if((temp+1.0) < 0.0)temp=-1.0;
    latangle[i]=(180.0/3.141592654)*acos(temp);
  }

  int imin,imax,imid;
  double min,max;

  max=min=latparam[0];
  imax=imin=0;

  for(i=0; i<3; i++){
    if(max <= latparam[i]){
      max=latparam[i];
      imax=i;
    }
    if(min > latparam[i]){
      min=latparam[i];
      imin=i;
    }
  }

  for(i=0; i<3; i++) if(i != imin && i !=imax)imid=i;

  //if all lattice parameters are equal length, numerical noise may cause imin=imax

  if(imin == imax)
    for(i=0; i<3; i++) if(i != imin && i !=imid)imax=i;

  permut[0]=imin;
  permut[1]=imid;
  permut[2]=imax;

  return;

}



//************************************************************
/*
  This function, given a lattice with vectors lat[3][3], finds the
  dimensions along the unit cell vectors such that a sphere of given radius
  fits within a uniform grid of 2dim[1]x2dim[2]x2dim[3] lattice points
  centered at the origin.

  The algorithm works by getting the normal (e.g. n1) to each pair of lattice
  vectors (e.g. a2, a3), scaling this normal to have length radius and
  then projecting this normal parallel to the a2,a3 plane onto the
  remaining lattice vector a1. This will tell us the number of a1 vectors
  needed to make a grid to encompass the sphere.
*/
//************************************************************

void lat_dimension(double lat[3][3], double radius, int dim[3]){
  int i,j,k;
  double inv_lat[3][3],normals[3][3],length[3];
  double frac_normals[3][3];

  //get the normals to pairs of lattice vectors of length radius

  for(i=0; i<3; i++){
    for(j=0; j<3; j++)normals[i][j]=lat[(i+1)%3][(j+1)%3]*lat[(i+2)%3][(j+2)%3]-
			lat[(i+1)%3][(j+2)%3]*lat[(i+2)%3][(j+1)%3];

    length[i]=0;
    for(j=0; j<3; j++)
      length[i]=length[i]+normals[i][j]*normals[i][j];
    length[i]=sqrt(length[i]);

    for(j=0; j<3; j++)normals[i][j]=radius*normals[i][j]/length[i];

  }

  //get the normals in the coordinates system of the lattice vectors

  inverse(lat,inv_lat);


  for(i=0; i<3; i++){
    for(j=0; j<3; j++){
      frac_normals[i][j]=0;
      for(k=0; k<3; k++)
	frac_normals[i][j]=frac_normals[i][j]+inv_lat[k][j]*normals[i][k];
    }
  }


  //the diagonals of frac_normal contain the dimensions of the lattice grid that
  //encompasses a sphere of radius = radius

  for(i=0; i<3; i++) dim[i]=(int)ceil(abs(frac_normals[i][i]));

  return;
}



//************************************************************

void conv_AtoB(double AtoB[3][3], double Acoord[3], double Bcoord[3]){
  int i,j;

  for(i=0; i<3; i++){
    Bcoord[i]=0.0;
    for(j=0; j<3 ; j++) Bcoord[i]=Bcoord[i]+AtoB[i][j]*Acoord[j];
  }
}



//************************************************************

double distance(atompos atom1,atompos atom2){
  double dist=0.0;
  for(int i=0; i<3; i++){
    dist=dist+(atom1.ccoord[i]-atom2.ccoord[i])*(atom1.ccoord[i]-atom2.ccoord[i]);
  }
  dist=sqrt(dist);
  return dist;
}


//************************************************************
//this routine starts from the first cluster in the orbit and
//generates all equivalent clusters by applying the factor_group
//symmetry operations

void get_equiv(orbit &orb, vector<sym_op> &op){
  int fg;
  cluster tclust1;


  if(orb.equiv.size() == 0){
    cout << "No cluster present \n";
    exit(1);
  }

  tclust1=orb.equiv[0];

  orb.equiv.clear();

  for(fg=0; fg < op.size(); fg++){
    cluster tclust2;
    tclust2=tclust1.apply_sym(op[fg]);
    within(tclust2);
    tclust2.get_cart(op[fg].FtoC);

    if(new_clust(tclust2,orb)){
      orb.equiv.push_back(tclust2);
    }

  }
}


//************************************************************
//checks to see whether a cluster clust already belongs to an orbit of clusters

bool new_clust(cluster clust, orbit &orb){
  int np,ne;


  if(orb.equiv.size() == 0) return true;

  if(clust.point.size() != orb.equiv[0].point.size()){
    cout << " comparing clusters with different number of points \n";
    cout << " the new_clust function was not meant for that \n";
    exit(1);
  };

  for(np=0; np<clust.point.size(); np++){
    cluster tclust;
    tclust=clust;
    within(tclust,np);
    for(ne=0; ne<orb.equiv.size(); ne++){
      if(compare(tclust,orb.equiv[ne])) return false;
    }
  }
  return true;
}



//************************************************************

bool new_clust(cluster clust, vector<orbit> &orbvec){
  int nc,non_match;

  if(orbvec.size() == 0) return true;

  if(clust.point.size() != orbvec[0].equiv[0].point.size()){
    cout << " comparing clusters with different number of points \n";
    cout << " the new_clust function was not meant for that \n";
    exit(1);
  };


  non_match=0;
  for(nc=0; nc<orbvec.size(); nc++){
    if(abs(orbvec[nc].equiv[0].max_leng - clust.max_leng) < 0.0001 &&
       abs(orbvec[nc].equiv[0].min_leng - clust.min_leng) < 0.0001){
      if(new_clust(clust,orbvec[nc]))non_match++;
    }
    else non_match++;
  }

  if(non_match == orbvec.size())return true;
  else return false;

}



//************************************************************
//this routine starts from the first cluster in the orbit and
//generates all equivalent clusters by applying the site_point_group
//symmetry operations

void get_loc_equiv(orbit &orb, vector<sym_op> &op){
  int g;
  cluster tclust1;


  if(orb.equiv.size() == 0){
    cout << "No cluster present \n";
    exit(1);
  }

  tclust1=orb.equiv[0];


  orb.equiv.clear();

  for(g=0; g < op.size(); g++){
    cluster tclust2;
    tclust2=tclust1.apply_sym(op[g]);
    tclust2.get_cart(op[g].FtoC);

    if(new_loc_clust(tclust2,orb)){
      orb.equiv.push_back(tclust2);
    }
  }
}


//************************************************************
//checks to see whether a cluster clust already belongs to an orbit of clusters

bool new_loc_clust(cluster clust, orbit orb){
  int np,ne;


  if(orb.equiv.size() == 0) return true;

  if(clust.point.size() != orb.equiv[0].point.size()){
    cout << " comparing clusters with different number of points \n";
    cout << " the new_clust function was not meant for that \n";
    exit(1);
  };

  for(ne=0; ne<orb.equiv.size(); ne++){
    if(compare(clust,orb.equiv[ne])) return false;
  }
  return true;
}



//************************************************************

bool new_loc_clust(cluster clust, vector<orbit> orbvec){
  int nc,non_match;


  if(orbvec.size() == 0) return true;

  if(clust.point.size() != orbvec[0].equiv[0].point.size()){
    cout << " comparing clusters with different number of points \n";
    cout << " the new_clust function was not meant for that \n";
    exit(1);
  };


  non_match=0;
  for(nc=0; nc<orbvec.size(); nc++){
    if(abs(orbvec[nc].equiv[0].max_leng-clust.max_leng) < 0.0001 &&
       abs(orbvec[nc].equiv[0].min_leng-clust.min_leng) < 0.0001){
      if(new_loc_clust(clust,orbvec[nc]))non_match++;
    }
    else non_match++;
  }

  if(non_match == orbvec.size())return true;
  else return false;

}


//************************************************************

void structure::read_species(){

  double temp;
  string tstring;
  vector <string> names;
  vector <double> masses;
  vector <double> magmoms;
  vector <double> Us;
  vector <double> Js;

  ifstream in;
  if(scandirectory(".","species")) in.open("species");
  else if(scandirectory(".","SPECIES")) in.open("SPECIES");
  else{
    cout << "No SPECIES file in the current directory \n";
    return;
  }
  if(!in){
    cout << "cannot open species\n";
    return;
  }

  while(tstring != "mass") {
    in >> tstring;
    if (tstring != "mass") names.push_back(tstring);
  }

  for(int i = 0; i < names.size(); i++) {
    in >>  temp;
    masses.push_back(temp);
  }
  in >> tstring;
  for(int i = 0; i < names.size(); i++) {
    in >>  temp;
    magmoms.push_back(temp);
  }
  in >> tstring;
  for(int i = 0; i < names.size(); i++) {
    in >>  temp;
    Us.push_back(temp);
  }
  in >> tstring;
  for(int i = 0; i < names.size(); i++) {
    in >>  temp;
    Js.push_back(temp);
  }


  for(int i=0;i<atom.size();i++) {
    for(int j=0;j<atom[i].compon.size();j++) {
      if(!(atom[i].compon[j].name.compare("Va") == 0)) {
	for(int k=0;k<names.size();k++) {
	  if(atom[i].compon[j].name.compare(names[k]) == 0) {
	    atom[i].compon[j].mass = masses[k];
	    atom[i].compon[j].magmom = magmoms[k];
	    atom[i].compon[j].U = Us[k];
	    atom[i].compon[j].J = Js[k];
	  }
	}
      }
    }
  }



  in.close();

  return;
}


//************************************************************

//************************************************************

void read_cspecs(vector<double> &max_radius){
  char buff[200];
  int dummy;
  double radius;

  ifstream in;
  if(scandirectory(".","cspecs")) in.open("cspecs");
  else if(scandirectory(".","CSPECS")) in.open("CSPECS");
  else cout << "No CSPECS file in the current directory \n";
  if(!in){
    cout << "cannot open cspecs\n";
    return;
  }

  radius=0.0;
  max_radius.push_back(radius);
  max_radius.push_back(radius);

  in.getline(buff,199);
  in.getline(buff,199);
  while(in >> dummy >> radius)
    max_radius.push_back(radius);

  in.close();

  return;
}


//************************************************************

void write_clust(multiplet clustiplet, string out_file){
  int num_clust=0;
  for(int i=1; i<clustiplet.orb.size(); i++){
    num_clust=num_clust+clustiplet.orb[i].size();
  }

  ofstream out;
  out.open(out_file.c_str());
  if(!out){
    cout << "cannot open " << out_file << "\n";
    return;
  }

  out << num_clust << "\n";
  num_clust=0;
  for(int i=1; i<clustiplet.orb.size(); i++){
    for(int j=0; j<clustiplet.orb[i].size(); j++){
      num_clust++;
      out << num_clust << "  " << clustiplet.orb[i][j].equiv[0].point.size() << "  "
	  << clustiplet.orb[i][j].equiv.size() << "  0  max length "
	  << clustiplet.orb[i][j].equiv[0].max_leng << "\n";
      clustiplet.orb[i][j].equiv[0].print(out);
    }
  }
  out.close();
}


//************************************************************

void write_fclust(multiplet clustiplet, string out_file){
  int num_clust=0;
  for(int i=1; i<clustiplet.orb.size(); i++){
    num_clust=num_clust+clustiplet.orb[i].size();
  }

  ofstream out;
  out.open(out_file.c_str());
  if(!out){
    cout << "cannot open " << out_file <<"\n";
    return;
  }

  out << num_clust << "\n";
  num_clust=0;
  for(int i=1; i<clustiplet.orb.size(); i++){
    for(int j=0; j<clustiplet.orb[i].size(); j++){
      num_clust++;
      out << " Orbit number " << num_clust << "\n";
      out << num_clust << "  " << clustiplet.orb[i][j].equiv[0].point.size() << "  "
	  << clustiplet.orb[i][j].equiv.size() << "  0  max length "
	  << clustiplet.orb[i][j].equiv[0].max_leng << "\n";
      clustiplet.orb[i][j].print(out);
    }
  }
  out.close();
}


//************************************************************

void write_scel(vector<structure> suplat){

  ofstream out;
  out.open("SCEL");
  if(!out){
    cout << "cannot open SCEL \n";
    return;
  }

  out << suplat.size() << "\n";
  for(int i=0; i<suplat.size(); i++){
    for(int j=0; j<3; j++){
      for(int k=0; k<3; k++){
	out.precision(5);out.width(5);
	out << suplat[i].slat[j][k] << " ";
      }
      out << "  ";
    }
    out << " volume = " << determinant(suplat[i].slat) << "\n";
  }
  out.close();
}


//************************************************************

void read_scel(vector<structure> &suplat, structure prim){
  suplat.clear();
  int num_scel;
  char buff[200];
  ifstream in;
  in.open("SCEL");
  if(!in){
    cout << "cannot open SCEL \n";
    return;
  }
  in >> num_scel;
  for(int n=0; n<num_scel; n++){
    structure tsuplat;
    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	in >> tsuplat.slat[i][j];
      }
    }
    in.getline(buff,199);
    tsuplat.scale=prim.scale;
    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	tsuplat.lat[i][j]=0.0;
	for(int k=0; k<3; k++){
	  tsuplat.lat[i][j]=tsuplat.lat[i][j]+tsuplat.slat[i][k]*prim.lat[k][j];
	}
      }
    }
    suplat.push_back(tsuplat);
  }
  in.close();
}



//************************************************************

bool update_bit(vector<int> max_bit, vector<int> &bit, int &last){

  //given a bit vector, this routine updates the bit by 1
  //if the maximum bit has been reached it returns false
  //last needs to be initialized as zero and bit needs to also be initialized as all zeros

  int bs=bit.size();

  if(last == 0){
    bit[0]++;
    for(int i=0; i<bs-1; i++){
      if(bit[i] !=0 && bit[i]%max_bit[i] == 0){
	bit[i+1]++;
	bit[i]=0;
      }
    }
    if(bit[bs-1] !=0 && bit[bs-1]%max_bit[bs-1] == 0){
      last=last+1;
      bit[bs-1]=0;
    }
    return true;
  }
  else{
    return false;
  }
}


//************************************************************

double ran0(int &idum){
  int IA=16807;
  int IM=2147483647;
  int IQ=127773;
  int IR=2836;
  int MASK=123459876;
  double AM=1.0/IM;

  //minimal random number generator of Park and Miller
  //returns uniform random deviate between 0.0 and 1.0
  //set or rest idum to any  integer value (except the
  //unlikely value MASK) to initialize the sequence: idum must
  //not be altered between calls for successive deviates
  //in a sequence

  int k;
  idum=idum^MASK;     //XOR the two integers
  k=idum/IQ;
  idum=IA*(idum-k*IQ)-IR*k;
  if(idum < 0) idum=idum+IM;
  double ran=AM*idum;
  idum=idum^MASK;
  return ran;

}


//************************************************************

//creates the shift vectors

void get_shift(atompos &atom, vector<atompos> basis){

  //first bring atom within the primitive unit cell
  //and document the translations needed for that

  atompos tatom=atom;

  for(int i=0; i<3; i++){
    atom.shift[i]=0;
    while(tatom.fcoord[i] < 0.0){
      atom.shift[i]=atom.shift[i]-1;
      tatom.fcoord[i]=tatom.fcoord[i]+1.0;
    }
    while(tatom.fcoord[i] > 0.99999){
      atom.shift[i]=atom.shift[i]+1;
      tatom.fcoord[i]=tatom.fcoord[i]-1.0;
    }
  }

  //then compare with all basis points and determine which one matches

  int nb=0;
  for(int na=0; na<basis.size(); na++){
    if(basis[na].compon.size() >= 2){
      nb++;
      if(compare(basis[na].fcoord,tatom.fcoord)){
        atom.shift[3]=nb-1;
        break;
      }
    }
  }

}




//************************************************************

// scans the directory 'dirname' to see whether 'filename' resides there
// by Qingchuan Xu

bool scandirectory(string dirname, string filename)
{
  bool exist=false;

  char ch;
  int  n;
  double e0;
  DIR *dir;
  struct dirent *entry;
  char path[PATH_MAX];
  dir = opendir(dirname.c_str());
  if (dir == NULL)
    {
      perror("Error opendir()");
      exit (0);
    }

  while ((entry = readdir(dir)) != NULL && !exist)
    {
      if (entry->d_type == DT_DIR)
        {
          if (strcmp(entry->d_name, ".")&& strcmp(entry->d_name, ".."))
            {
	      if(!fnmatch(filename.c_str(), entry->d_name, 0)) exist=true;
            }
        }
      else if (entry->d_type == DT_REG)
        {
          if (!fnmatch(filename.c_str(), entry->d_name, 0)) exist=true;
        }
    }
  closedir(dir);
  return exist;

}


//************************************************************

// reads the OSZICAR in dirname and extracts the final energy
// by Qingchuan Xu

bool read_oszicar(string dirname, double& e0)
{
  static bool exist=false;
  char ch;
  int  n;
  ifstream readfrom;
  DIR *dir;
  struct dirent *entry;
  char path[PATH_MAX];
  char path1[PATH_MAX];
  dir = opendir(dirname.c_str());
  if (dir == NULL)
    {
      perror("Error opendir()");
      return 0;
    }

  //stop_flag is used to override original recursive behavior (read OSZICAR of lowest level directory).
  //remove all reference to restore previous behavior
  bool stop_flag=true;
  while ((entry = readdir(dir)) != NULL && stop_flag)
    {
      if (entry->d_type == DT_DIR && !stop_flag)
        {
          if (strcmp(entry->d_name, ".")&& strcmp(entry->d_name, ".."))
            {
              snprintf(path, (size_t) PATH_MAX, "%s/%s", dirname.c_str(),entry->d_name);
              read_oszicar(path, e0);
            }
        }
      else if (entry->d_type == DT_REG || stop_flag)
        {
          if (!fnmatch("OSZICAR", entry->d_name, 0))
            {
	      stop_flag=false;
              exist = true;
              snprintf(path1, (size_t) PATH_MAX, "%s/%s", dirname.c_str(), entry->d_name);
              readfrom.open(path1);
	      do
		{
		  readfrom.get(ch);
		  if (ch=='F')
		    {n=0;
		      do{readfrom.get(ch);
			if(ch=='=')
			  n=n+1;
		      }
		      while (n<2);
		      readfrom>>e0;
		    }
		}
	      while (!readfrom.eof());
	      readfrom.close();

            }
        }
    }
  closedir(dir);
  return exist;
}
// *******************************************************************
// *******************************************************************
bool read_oszicar(string dirname, double& e0, int& count)
{
  static bool exist=false;
  char ch;
  int  n;
  ifstream readfrom;
  DIR *dir;
  struct dirent *entry;
  char path[PATH_MAX];
  char path1[PATH_MAX];
  dir = opendir(dirname.c_str());
  if (dir == NULL)
    {
      perror("Error opendir()");
      return 0;
    }
  bool stop_flag=true;  //This is just being used to override original recursive behavior.  remove all reference to restore previous behavior
  while ((entry = readdir(dir)) != NULL && stop_flag)
    {
      if (entry->d_type == DT_DIR && !stop_flag)
        {
	  if (strcmp(entry->d_name, ".")&& strcmp(entry->d_name, ".."))
            {
	      snprintf(path, (size_t) PATH_MAX, "%s/%s", dirname.c_str(),entry->d_name);
	      read_oszicar(path, e0, count);
            }
        }
      else if (entry->d_type == DT_REG || stop_flag)
        {
	  if (!fnmatch("OSZICAR", entry->d_name, 0))
            {
	      stop_flag=false;
	      exist = true;
	      snprintf(path1, (size_t) PATH_MAX, "%s/%s", dirname.c_str(), entry->d_name);
	      readfrom.open(path1);
	      count=0;
	      do
		{
		  readfrom.get(ch);
		  if (ch=='F')
		    {count++;
		      n=0;
		      do{readfrom.get(ch);
			if(ch=='=')
			  n=n+1;
		      }
		      while (n<2);
		      readfrom>>e0;
		    }
		}
	      while (!readfrom.eof());
	      readfrom.close();

            }
        }
    }
  closedir(dir);
  return exist;
}
// *******************************************************************

bool read_mc_input(string cond_file, int &n_pass, int &n_equil_pass, int &nx, int &ny, int &nz, double &Tinit, double &Tmin, double &Tmax, double &Tinc, chempot &muinit, chempot &mu_min, chempot &mu_max, vector<chempot> &muinc, int &xyz_step, int &corr_flag, int &temp_chem){

  ifstream in;
  in.open(cond_file.c_str());
  if(!in){
    cout << "cannot open " << cond_file << "\n";
    return false;
  }

  char buff[200];
  in >> n_pass;
  in.getline(buff,199);
  in >> n_equil_pass;
  in.getline(buff,199);
  in >> nx;
  in >> ny;
  in >> nz;
  in.getline(buff,199);
  in >> Tinit;
  in.getline(buff,199);
  in >> Tmin;
  in.getline(buff,199);
  in >> Tmax;
  in.getline(buff,199);
  in >> Tinc;
  in.getline(buff,199);


  in.getline(buff,199);
  int buffcount=0;
  string tspec;
  double tdouble;
  while(buff[buffcount]!='!'){


    while((buff[buffcount]>='A'&&buff[buffcount]<='Z')||(buff[buffcount]>='a'&&buff[buffcount]<='z')){
      tspec.push_back(buff[buffcount]);
      buffcount++;
    }
    if(buff[buffcount]>='-'&&buff[buffcount]<='9'){
      string tstring;
      while(buff[buffcount]>='-'&&buff[buffcount]<='9'){
	tstring.push_back(buff[buffcount]);
	buffcount++;
      }
      tdouble=strtod(tstring.c_str(), NULL);

      for(int ii=0; ii<muinit.compon.size(); ii++){
	for(int jj=0; jj<muinit.compon[ii].size(); jj++){
	  if(!muinit.compon[ii][jj].name.compare(tspec)){
	    //	    muinit.m[ii].erase(jj);
	    muinit.m[ii][jj]=tdouble;
	  }
	}
      }
      tspec.clear();
    }
    else buffcount++;
  }
  buffcount=0;

  in.getline(buff,199);
  while(buff[buffcount]!='!'){

    while((buff[buffcount]>='A'&&buff[buffcount]<='Z')||(buff[buffcount]>='a'&&buff[buffcount]<='z')){
      tspec.push_back(buff[buffcount]);
      buffcount++;
    }
    if(buff[buffcount]>='-'&&buff[buffcount]<='9'){
      string tstring;
      while(buff[buffcount]>='-'&&buff[buffcount]<='9'){
	tstring.push_back(buff[buffcount]);
	buffcount++;
      }
      tdouble=strtod(tstring.c_str(), NULL);
      for(int ii=0; ii<mu_min.compon.size(); ii++){
	for(int jj=0; jj<mu_min.compon[ii].size(); jj++){
	  if(!mu_min.compon[ii][jj].name.compare(tspec)){
	    //	    mu_min.m[ii].erase(jj);
	    mu_min.m[ii][jj]=tdouble;
	  }
	}
      }
      tspec.clear();
    }
    else buffcount++;
  }
  buffcount=0;

  in.getline(buff,199);

  while(buff[buffcount]!='!'){

    while((buff[buffcount]>='A'&&buff[buffcount]<='Z')||(buff[buffcount]>='a'&&buff[buffcount]<='z')){
      tspec.push_back(buff[buffcount]);
      buffcount++;
    }
    if(buff[buffcount]>='-'&&buff[buffcount]<='9'){
      string tstring;
      while(buff[buffcount]>='-'&&buff[buffcount]<='9'){
	tstring.push_back(buff[buffcount]);
	buffcount++;
      }
      tdouble=strtod(tstring.c_str(), NULL);
      for(int ii=0; ii<mu_max.compon.size(); ii++){
	for(int jj=0; jj<mu_max.compon[ii].size(); jj++){
	  if(!mu_max.compon[ii][jj].name.compare(tspec)){
	    //	    mu_max.m[ii].erase(jj);
	    mu_max.m[ii][jj]=tdouble;
	  }
	}
      }
      tspec.clear();
    }
    else buffcount++;
  }

  buffcount=0;


  in.getline(buff,199);

  while(buff[buffcount]!='!'){

    while((buff[buffcount]>='A'&&buff[buffcount]<='Z')||(buff[buffcount]>='a'&&buff[buffcount]<='z')){
      tspec.push_back(buff[buffcount]);
      buffcount++;
    }
    if(buff[buffcount]>='-'&&buff[buffcount]<='9'){
      string tstring;
      while(buff[buffcount]>='-'&&buff[buffcount]<='9'){
	tstring.push_back(buff[buffcount]);
	buffcount++;
      }
      tdouble=strtod(tstring.c_str(), NULL);
      chempot tmu;
      tmu.initialize(muinit.compon);
      for(int ii=0; ii<tmu.compon.size(); ii++){
	for(int jj=0; jj<tmu.compon[ii].size(); jj++){
	  if(!tmu.compon[ii][jj].name.compare(tspec)){
	    //	    muinc.m[ii].erase(jj);
	    tmu.m[ii][jj]=tdouble;
	    muinc.push_back(tmu);
	  }
	}
      }
      tspec.clear();
    }
    else buffcount++;
  }
  buffcount=0;
  in >> xyz_step;
  in.getline(buff,199);
  in >> corr_flag;
  in.getline(buff,199);
  in >> temp_chem;
  in.getline(buff,199);
  in.close();
  return true;
}

double Monte_Carlo::pointenergy(int i, int j, int k, int b){
  double energy = 0.0;
  int l; 
  if(b == 0){
     l=index(i,j,k,0);
     int p00=mcL[l];
     l=index(i,j,k,0);
     int p01=mcL[l]*mcL[l];
     l=index(i,j-1,k-1,2);
     int p10=mcL[l];
     l=index(i,j-1,k-1,2);
     int p11=mcL[l]*mcL[l];
     l=index(i,j,k,1);
     int p20=mcL[l];
     l=index(i,j,k,1);
     int p21=mcL[l]*mcL[l];
     l=index(i,j,k-1,1);
     int p30=mcL[l];
     l=index(i,j,k-1,1);
     int p31=mcL[l]*mcL[l];
     l=index(i,j-1,k,1);
     int p40=mcL[l];
     l=index(i,j-1,k,1);
     int p41=mcL[l]*mcL[l];
     l=index(i-1,j,k-1,2);
     int p50=mcL[l];
     l=index(i-1,j,k-1,2);
     int p51=mcL[l]*mcL[l];
     l=index(i-1,j-1,k,2);
     int p60=mcL[l];
     l=index(i-1,j-1,k,2);
     int p61=mcL[l]*mcL[l];
     l=index(i-1,j-1,k-1,2);
     int p70=mcL[l];
     l=index(i-1,j-1,k-1,2);
     int p71=mcL[l]*mcL[l];
     l=index(i-1,j,k,1);
     int p80=mcL[l];
     l=index(i-1,j,k,1);
     int p81=mcL[l]*mcL[l];
     l=index(i+1,j,k,0);
     int p90=mcL[l];
     l=index(i+1,j,k,0);
     int p91=mcL[l]*mcL[l];
     l=index(i-1,j,k,0);
     int p100=mcL[l];
     l=index(i-1,j,k,0);
     int p101=mcL[l]*mcL[l];
     l=index(i+1,j,k-1,0);
     int p110=mcL[l];
     l=index(i+1,j,k-1,0);
     int p111=mcL[l]*mcL[l];
     l=index(i-1,j,k+1,0);
     int p120=mcL[l];
     l=index(i-1,j,k+1,0);
     int p121=mcL[l]*mcL[l];
     l=index(i+1,j-1,k,0);
     int p130=mcL[l];
     l=index(i+1,j-1,k,0);
     int p131=mcL[l]*mcL[l];
     l=index(i-1,j+1,k,0);
     int p140=mcL[l];
     l=index(i-1,j+1,k,0);
     int p141=mcL[l]*mcL[l];
     l=index(i,j+1,k,0);
     int p150=mcL[l];
     l=index(i,j+1,k,0);
     int p151=mcL[l]*mcL[l];
     l=index(i,j-1,k,0);
     int p160=mcL[l];
     l=index(i,j-1,k,0);
     int p161=mcL[l]*mcL[l];
     l=index(i,j+1,k-1,0);
     int p170=mcL[l];
     l=index(i,j+1,k-1,0);
     int p171=mcL[l]*mcL[l];
     l=index(i,j-1,k+1,0);
     int p180=mcL[l];
     l=index(i,j-1,k+1,0);
     int p181=mcL[l]*mcL[l];
     l=index(i,j,k+1,0);
     int p190=mcL[l];
     l=index(i,j,k+1,0);
     int p191=mcL[l]*mcL[l];
     l=index(i,j,k-1,0);
     int p200=mcL[l];
     l=index(i,j,k-1,0);
     int p201=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p210=mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p211=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,0);
     int p220=mcL[l];
     l=index(i-1,j-1,k+1,0);
     int p221=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p230=mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p231=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,0);
     int p240=mcL[l];
     l=index(i-1,j+1,k-1,0);
     int p241=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p250=mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p251=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,0);
     int p260=mcL[l];
     l=index(i+1,j-1,k-1,0);
     int p261=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,2);
     int p270=mcL[l];
     l=index(i+1,j-1,k-1,2);
     int p271=mcL[l]*mcL[l];
     l=index(i+1,j,k,1);
     int p280=mcL[l];
     l=index(i+1,j,k,1);
     int p281=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-2,2);
     int p290=mcL[l];
     l=index(i+1,j-1,k-2,2);
     int p291=mcL[l]*mcL[l];
     l=index(i+1,j,k-2,1);
     int p300=mcL[l];
     l=index(i+1,j,k-2,1);
     int p301=mcL[l]*mcL[l];
     l=index(i+1,j-2,k-1,2);
     int p310=mcL[l];
     l=index(i+1,j-2,k-1,2);
     int p311=mcL[l]*mcL[l];
     l=index(i+1,j-2,k,1);
     int p320=mcL[l];
     l=index(i+1,j-2,k,1);
     int p321=mcL[l]*mcL[l];
     l=index(i,j+1,k,1);
     int p330=mcL[l];
     l=index(i,j+1,k,1);
     int p331=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,2);
     int p340=mcL[l];
     l=index(i-1,j+1,k-1,2);
     int p341=mcL[l]*mcL[l];
     l=index(i,j+1,k-2,1);
     int p350=mcL[l];
     l=index(i,j+1,k-2,1);
     int p351=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-2,2);
     int p360=mcL[l];
     l=index(i-1,j+1,k-2,2);
     int p361=mcL[l]*mcL[l];
     l=index(i,j,k+1,1);
     int p370=mcL[l];
     l=index(i,j,k+1,1);
     int p371=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,2);
     int p380=mcL[l];
     l=index(i-1,j-1,k+1,2);
     int p381=mcL[l]*mcL[l];
     l=index(i,j,k-2,1);
     int p390=mcL[l];
     l=index(i,j,k-2,1);
     int p391=mcL[l]*mcL[l];
     l=index(i-1,j-1,k-2,2);
     int p400=mcL[l];
     l=index(i-1,j-1,k-2,2);
     int p401=mcL[l]*mcL[l];
     l=index(i,j-2,k+1,1);
     int p410=mcL[l];
     l=index(i,j-2,k+1,1);
     int p411=mcL[l]*mcL[l];
     l=index(i-1,j-2,k+1,2);
     int p420=mcL[l];
     l=index(i-1,j-2,k+1,2);
     int p421=mcL[l]*mcL[l];
     l=index(i,j-2,k,1);
     int p430=mcL[l];
     l=index(i,j-2,k,1);
     int p431=mcL[l]*mcL[l];
     l=index(i-1,j-2,k-1,2);
     int p440=mcL[l];
     l=index(i-1,j-2,k-1,2);
     int p441=mcL[l]*mcL[l];
     l=index(i-2,j+1,k-1,2);
     int p450=mcL[l];
     l=index(i-2,j+1,k-1,2);
     int p451=mcL[l]*mcL[l];
     l=index(i-2,j+1,k,1);
     int p460=mcL[l];
     l=index(i-2,j+1,k,1);
     int p461=mcL[l]*mcL[l];
     l=index(i-2,j-1,k+1,2);
     int p470=mcL[l];
     l=index(i-2,j-1,k+1,2);
     int p471=mcL[l]*mcL[l];
     l=index(i-2,j,k+1,1);
     int p480=mcL[l];
     l=index(i-2,j,k+1,1);
     int p481=mcL[l]*mcL[l];
     l=index(i-2,j-1,k-1,2);
     int p490=mcL[l];
     l=index(i-2,j-1,k-1,2);
     int p491=mcL[l]*mcL[l];
     l=index(i-2,j,k,1);
     int p500=mcL[l];
     l=index(i-2,j,k,1);
     int p501=mcL[l]*mcL[l];
     l=index(i+2,j,k-1,0);
     int p510=mcL[l];
     l=index(i+2,j,k-1,0);
     int p511=mcL[l]*mcL[l];
     l=index(i-2,j,k+1,0);
     int p520=mcL[l];
     l=index(i-2,j,k+1,0);
     int p521=mcL[l]*mcL[l];
     l=index(i+2,j-1,k,0);
     int p530=mcL[l];
     l=index(i+2,j-1,k,0);
     int p531=mcL[l]*mcL[l];
     l=index(i-2,j+1,k,0);
     int p540=mcL[l];
     l=index(i-2,j+1,k,0);
     int p541=mcL[l]*mcL[l];
     l=index(i+1,j+1,k,0);
     int p550=mcL[l];
     l=index(i+1,j+1,k,0);
     int p551=mcL[l]*mcL[l];
     l=index(i-1,j-1,k,0);
     int p560=mcL[l];
     l=index(i-1,j-1,k,0);
     int p561=mcL[l]*mcL[l];
     l=index(i+1,j,k+1,0);
     int p570=mcL[l];
     l=index(i+1,j,k+1,0);
     int p571=mcL[l]*mcL[l];
     l=index(i-1,j,k-1,0);
     int p580=mcL[l];
     l=index(i-1,j,k-1,0);
     int p581=mcL[l]*mcL[l];
     l=index(i+2,j-1,k-1,0);
     int p590=mcL[l];
     l=index(i+2,j-1,k-1,0);
     int p591=mcL[l]*mcL[l];
     l=index(i-2,j+1,k+1,0);
     int p600=mcL[l];
     l=index(i-2,j+1,k+1,0);
     int p601=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-2,0);
     int p610=mcL[l];
     l=index(i+1,j+1,k-2,0);
     int p611=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+2,0);
     int p620=mcL[l];
     l=index(i-1,j-1,k+2,0);
     int p621=mcL[l]*mcL[l];
     l=index(i-1,j,k+2,0);
     int p630=mcL[l];
     l=index(i-1,j,k+2,0);
     int p631=mcL[l]*mcL[l];
     l=index(i+1,j,k-2,0);
     int p640=mcL[l];
     l=index(i+1,j,k-2,0);
     int p641=mcL[l]*mcL[l];
     l=index(i+1,j-2,k+1,0);
     int p650=mcL[l];
     l=index(i+1,j-2,k+1,0);
     int p651=mcL[l]*mcL[l];
     l=index(i-1,j+2,k-1,0);
     int p660=mcL[l];
     l=index(i-1,j+2,k-1,0);
     int p661=mcL[l]*mcL[l];
     l=index(i-1,j+2,k,0);
     int p670=mcL[l];
     l=index(i-1,j+2,k,0);
     int p671=mcL[l]*mcL[l];
     l=index(i+1,j-2,k,0);
     int p680=mcL[l];
     l=index(i+1,j-2,k,0);
     int p681=mcL[l]*mcL[l];
     l=index(i,j+2,k-1,0);
     int p690=mcL[l];
     l=index(i,j+2,k-1,0);
     int p691=mcL[l]*mcL[l];
     l=index(i,j-2,k+1,0);
     int p700=mcL[l];
     l=index(i,j-2,k+1,0);
     int p701=mcL[l]*mcL[l];
     l=index(i,j+1,k+1,0);
     int p710=mcL[l];
     l=index(i,j+1,k+1,0);
     int p711=mcL[l]*mcL[l];
     l=index(i,j-1,k-1,0);
     int p720=mcL[l];
     l=index(i,j-1,k-1,0);
     int p721=mcL[l]*mcL[l];
     l=index(i,j+1,k-2,0);
     int p730=mcL[l];
     l=index(i,j+1,k-2,0);
     int p731=mcL[l]*mcL[l];
     l=index(i,j-1,k+2,0);
     int p740=mcL[l];
     l=index(i,j-1,k+2,0);
     int p741=mcL[l]*mcL[l];
     l=index(i,j,k,2);
     int p750=mcL[l];
     l=index(i,j,k,2);
     int p751=mcL[l]*mcL[l];
     l=index(i+2,j-1,k-1,1);
     int p760=mcL[l];
     l=index(i+2,j-1,k-1,1);
     int p761=mcL[l]*mcL[l];
     l=index(i,j,k-3,2);
     int p770=mcL[l];
     l=index(i,j,k-3,2);
     int p771=mcL[l]*mcL[l];
     l=index(i,j-3,k,2);
     int p780=mcL[l];
     l=index(i,j-3,k,2);
     int p781=mcL[l]*mcL[l];
     l=index(i-1,j+2,k-1,1);
     int p790=mcL[l];
     l=index(i-1,j+2,k-1,1);
     int p791=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+2,1);
     int p800=mcL[l];
     l=index(i-1,j-1,k+2,1);
     int p801=mcL[l]*mcL[l];
     l=index(i-1,j-1,k-1,1);
     int p810=mcL[l];
     l=index(i-1,j-1,k-1,1);
     int p811=mcL[l]*mcL[l];
     l=index(i-3,j,k,2);
     int p820=mcL[l];
     l=index(i-3,j,k,2);
     int p821=mcL[l]*mcL[l];
     l=index(i+2,j,k,0);
     int p830=mcL[l];
     l=index(i+2,j,k,0);
     int p831=mcL[l]*mcL[l];
     l=index(i-2,j,k,0);
     int p840=mcL[l];
     l=index(i-2,j,k,0);
     int p841=mcL[l]*mcL[l];
     l=index(i+2,j,k-2,0);
     int p850=mcL[l];
     l=index(i+2,j,k-2,0);
     int p851=mcL[l]*mcL[l];
     l=index(i-2,j,k+2,0);
     int p860=mcL[l];
     l=index(i-2,j,k+2,0);
     int p861=mcL[l]*mcL[l];
     l=index(i+2,j-2,k,0);
     int p870=mcL[l];
     l=index(i+2,j-2,k,0);
     int p871=mcL[l]*mcL[l];
     l=index(i-2,j+2,k,0);
     int p880=mcL[l];
     l=index(i-2,j+2,k,0);
     int p881=mcL[l]*mcL[l];
     l=index(i,j+2,k,0);
     int p890=mcL[l];
     l=index(i,j+2,k,0);
     int p891=mcL[l]*mcL[l];
     l=index(i,j-2,k,0);
     int p900=mcL[l];
     l=index(i,j-2,k,0);
     int p901=mcL[l]*mcL[l];
     l=index(i,j+2,k-2,0);
     int p910=mcL[l];
     l=index(i,j+2,k-2,0);
     int p911=mcL[l]*mcL[l];
     l=index(i,j-2,k+2,0);
     int p920=mcL[l];
     l=index(i,j-2,k+2,0);
     int p921=mcL[l]*mcL[l];
     l=index(i,j,k+2,0);
     int p930=mcL[l];
     l=index(i,j,k+2,0);
     int p931=mcL[l]*mcL[l];
     l=index(i,j,k-2,0);
     int p940=mcL[l];
     l=index(i,j,k-2,0);
     int p941=mcL[l]*mcL[l];
     l=index(i-2,j,k,2);
     int p950=mcL[l];
     l=index(i-2,j,k,2);
     int p951=mcL[l]*mcL[l];
     l=index(i-1,j,k,2);
     int p960=mcL[l];
     l=index(i-1,j,k,2);
     int p961=mcL[l]*mcL[l];
     l=index(i,j-1,k-1,1);
     int p970=mcL[l];
     l=index(i,j-1,k-1,1);
     int p971=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,1);
     int p980=mcL[l];
     l=index(i+1,j-1,k-1,1);
     int p981=mcL[l]*mcL[l];
     l=index(i-2,j,k-1,2);
     int p990=mcL[l];
     l=index(i-2,j,k-1,2);
     int p991=mcL[l]*mcL[l];
     l=index(i-1,j,k-2,2);
     int p1000=mcL[l];
     l=index(i-1,j,k-2,2);
     int p1001=mcL[l]*mcL[l];
     l=index(i,j-1,k+1,1);
     int p1010=mcL[l];
     l=index(i,j-1,k+1,1);
     int p1011=mcL[l]*mcL[l];
     l=index(i+1,j-1,k,1);
     int p1020=mcL[l];
     l=index(i+1,j-1,k,1);
     int p1021=mcL[l]*mcL[l];
     l=index(i-2,j-1,k,2);
     int p1030=mcL[l];
     l=index(i-2,j-1,k,2);
     int p1031=mcL[l]*mcL[l];
     l=index(i-1,j-2,k,2);
     int p1040=mcL[l];
     l=index(i-1,j-2,k,2);
     int p1041=mcL[l]*mcL[l];
     l=index(i,j+1,k-1,1);
     int p1050=mcL[l];
     l=index(i,j+1,k-1,1);
     int p1051=mcL[l]*mcL[l];
     l=index(i+1,j,k-1,1);
     int p1060=mcL[l];
     l=index(i+1,j,k-1,1);
     int p1061=mcL[l]*mcL[l];
     l=index(i-1,j,k-1,1);
     int p1070=mcL[l];
     l=index(i-1,j,k-1,1);
     int p1071=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,1);
     int p1080=mcL[l];
     l=index(i-1,j+1,k-1,1);
     int p1081=mcL[l]*mcL[l];
     l=index(i,j-2,k,2);
     int p1090=mcL[l];
     l=index(i,j-2,k,2);
     int p1091=mcL[l]*mcL[l];
     l=index(i,j-1,k,2);
     int p1100=mcL[l];
     l=index(i,j-1,k,2);
     int p1101=mcL[l]*mcL[l];
     l=index(i-1,j,k+1,1);
     int p1110=mcL[l];
     l=index(i-1,j,k+1,1);
     int p1111=mcL[l]*mcL[l];
     l=index(i-1,j+1,k,1);
     int p1120=mcL[l];
     l=index(i-1,j+1,k,1);
     int p1121=mcL[l]*mcL[l];
     l=index(i,j-2,k-1,2);
     int p1130=mcL[l];
     l=index(i,j-2,k-1,2);
     int p1131=mcL[l]*mcL[l];
     l=index(i,j-1,k-2,2);
     int p1140=mcL[l];
     l=index(i,j-1,k-2,2);
     int p1141=mcL[l]*mcL[l];
     l=index(i-1,j-1,k,1);
     int p1150=mcL[l];
     l=index(i-1,j-1,k,1);
     int p1151=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,1);
     int p1160=mcL[l];
     l=index(i-1,j-1,k+1,1);
     int p1161=mcL[l]*mcL[l];
     l=index(i,j,k-2,2);
     int p1170=mcL[l];
     l=index(i,j,k-2,2);
     int p1171=mcL[l]*mcL[l];
     l=index(i,j,k-1,2);
     int p1180=mcL[l];
     l=index(i,j,k-1,2);
     int p1181=mcL[l]*mcL[l];
     l=index(i-1,j-2,k+1,1);
     int p1190=mcL[l];
     l=index(i-1,j-2,k+1,1);
     int p1191=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-2,1);
     int p1200=mcL[l];
     l=index(i-1,j+1,k-2,1);
     int p1201=mcL[l]*mcL[l];
     l=index(i-2,j-2,k+1,2);
     int p1210=mcL[l];
     l=index(i-2,j-2,k+1,2);
     int p1211=mcL[l]*mcL[l];
     l=index(i-2,j+1,k-2,2);
     int p1220=mcL[l];
     l=index(i-2,j+1,k-2,2);
     int p1221=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,1);
     int p1230=mcL[l];
     l=index(i-1,j+1,k+1,1);
     int p1231=mcL[l]*mcL[l];
     l=index(i-2,j-2,k,2);
     int p1240=mcL[l];
     l=index(i-2,j-2,k,2);
     int p1241=mcL[l]*mcL[l];
     l=index(i-2,j+1,k,2);
     int p1250=mcL[l];
     l=index(i-2,j+1,k,2);
     int p1251=mcL[l]*mcL[l];
     l=index(i-2,j,k-2,2);
     int p1260=mcL[l];
     l=index(i-2,j,k-2,2);
     int p1261=mcL[l]*mcL[l];
     l=index(i-2,j,k+1,2);
     int p1270=mcL[l];
     l=index(i-2,j,k+1,2);
     int p1271=mcL[l]*mcL[l];
     l=index(i-2,j-1,k+1,1);
     int p1280=mcL[l];
     l=index(i-2,j-1,k+1,1);
     int p1281=mcL[l]*mcL[l];
     l=index(i+1,j-2,k-2,2);
     int p1290=mcL[l];
     l=index(i+1,j-2,k-2,2);
     int p1291=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-2,1);
     int p1300=mcL[l];
     l=index(i+1,j-1,k-2,1);
     int p1301=mcL[l]*mcL[l];
     l=index(i+1,j-2,k,2);
     int p1310=mcL[l];
     l=index(i+1,j-2,k,2);
     int p1311=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,1);
     int p1320=mcL[l];
     l=index(i+1,j-1,k+1,1);
     int p1321=mcL[l]*mcL[l];
     l=index(i-2,j+1,k-1,1);
     int p1330=mcL[l];
     l=index(i-2,j+1,k-1,1);
     int p1331=mcL[l]*mcL[l];
     l=index(i+1,j-2,k-1,1);
     int p1340=mcL[l];
     l=index(i+1,j-2,k-1,1);
     int p1341=mcL[l]*mcL[l];
     l=index(i-2,j+1,k+1,1);
     int p1350=mcL[l];
     l=index(i-2,j+1,k+1,1);
     int p1351=mcL[l]*mcL[l];
     l=index(i+1,j-2,k+1,1);
     int p1360=mcL[l];
     l=index(i+1,j-2,k+1,1);
     int p1361=mcL[l]*mcL[l];
     l=index(i+1,j,k-2,2);
     int p1370=mcL[l];
     l=index(i+1,j,k-2,2);
     int p1371=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,1);
     int p1380=mcL[l];
     l=index(i+1,j+1,k-1,1);
     int p1381=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-2,1);
     int p1390=mcL[l];
     l=index(i+1,j+1,k-2,1);
     int p1391=mcL[l]*mcL[l];
     l=index(i,j-2,k-2,2);
     int p1400=mcL[l];
     l=index(i,j-2,k-2,2);
     int p1401=mcL[l]*mcL[l];
     l=index(i,j-2,k+1,2);
     int p1410=mcL[l];
     l=index(i,j-2,k+1,2);
     int p1411=mcL[l]*mcL[l];
     l=index(i,j+1,k-2,2);
     int p1420=mcL[l];
     l=index(i,j+1,k-2,2);
     int p1421=mcL[l]*mcL[l];

     energy = energy-7.46574*(p01)+0.981589*(p00)-0.0346896*(p11*p00+p21*p00+p31*p00+p41*p00+p51*p00+p61*p00+p71*p00+p81*p00)-0.318022*(p10*p01+p20*p01+p30*p01+p40*p01+p50*p01+p60*p01+p70*p01+p80*p01)+1.35905*(p11*p01+p21*p01+p31*p01+p41*p01+p51*p01+p61*p01+p71*p01+p81*p01)+0.168277*(p10*p00+p20*p00+p30*p00+p40*p00+p50*p00+p60*p00+p70*p00+p80*p00)-0.0698276*(p01*p90+p101*p00+p01*p100+p91*p00+p01*p110+p121*p00+p01*p120+p111*p00+p01*p130+p141*p00+p01*p140+p131*p00+p01*p150+p161*p00+p01*p160+p151*p00+p01*p170+p181*p00+p01*p180+p171*p00+p01*p190+p201*p00+p01*p200+p191*p00)-0.237998*(p01*p210+p221*p00+p01*p220+p211*p00+p01*p230+p241*p00+p01*p240+p231*p00+p01*p250+p261*p00+p01*p260+p251*p00)-0.00322933*(p271*p01+p281*p01+p291*p01+p301*p01+p311*p01+p321*p01+p331*p01+p341*p01+p351*p01+p361*p01+p371*p01+p381*p01+p391*p01+p401*p01+p411*p01+p421*p01+p431*p01+p441*p01+p451*p01+p461*p01+p471*p01+p481*p01+p491*p01+p501*p01)-0.0399088*(p01*p511+p521*p01+p01*p531+p541*p01+p01*p551+p561*p01+p01*p571+p581*p01+p01*p591+p601*p01+p01*p611+p621*p01+p01*p631+p641*p01+p01*p651+p661*p01+p01*p671+p681*p01+p01*p691+p701*p01+p01*p711+p721*p01+p01*p731+p741*p01)-0.0144782*(p00*p510+p520*p00+p00*p530+p540*p00+p00*p550+p560*p00+p00*p570+p580*p00+p00*p590+p600*p00+p00*p610+p620*p00+p00*p630+p640*p00+p00*p650+p660*p00+p00*p670+p680*p00+p00*p690+p700*p00+p00*p710+p720*p00+p00*p730+p740*p00)-0.0766549*(p751*p00+p761*p00+p771*p00+p781*p00+p791*p00+p801*p00+p811*p00+p821*p00)+0.00623218*(p751*p01+p761*p01+p771*p01+p781*p01+p791*p01+p801*p01+p811*p01+p821*p01)+0.0842672*(p750*p00+p760*p00+p770*p00+p780*p00+p790*p00+p800*p00+p810*p00+p820*p00)-0.0170584*(p00*p830+p840*p00+p00*p850+p860*p00+p00*p870+p880*p00+p00*p890+p900*p00+p00*p910+p920*p00+p00*p930+p940*p00)-0.00329238*(p01*p100*p120+p91*p00*p190+p111*p200*p00+p01*p200*p110+p191*p00*p90+p121*p100*p00+p01*p190*p90+p201*p00*p110+p101*p120*p00+p01*p100*p140+p91*p00*p150+p131*p160*p00+p01*p160*p130+p151*p00*p90+p141*p100*p00+p01*p150*p90+p161*p00*p130+p101*p140*p00+p01*p100*p160+p91*p00*p130+p151*p140*p00+p01*p140*p150+p131*p00*p90+p161*p100*p00+p01*p130*p90+p141*p00*p150+p101*p160*p00+p01*p100*p200+p91*p00*p110+p191*p120*p00+p01*p120*p190+p111*p00*p90+p201*p100*p00+p01*p110*p90+p121*p00*p190+p101*p200*p00+p01*p120*p140+p111*p00*p170+p131*p180*p00+p01*p180*p130+p171*p00*p110+p141*p120*p00+p01*p170*p110+p181*p00*p130+p121*p140*p00+p01*p120*p180+p111*p00*p130+p171*p140*p00+p01*p140*p170+p131*p00*p110+p181*p120*p00+p01*p130*p110+p141*p00*p170+p121*p180*p00+p01*p160*p180+p151*p00*p190+p171*p200*p00+p01*p200*p170+p191*p00*p150+p181*p160*p00+p01*p190*p150+p201*p00*p170+p161*p180*p00+p01*p160*p200+p151*p00*p170+p191*p180*p00+p01*p180*p190+p171*p00*p150+p201*p160*p00+p01*p170*p150+p181*p00*p190+p161*p200*p00)+0.00574111*(p00*p100*p120+p90*p00*p190+p110*p200*p00+p00*p100*p140+p90*p00*p150+p130*p160*p00+p00*p100*p160+p90*p00*p130+p150*p140*p00+p00*p100*p200+p90*p00*p110+p190*p120*p00+p00*p120*p140+p110*p00*p170+p130*p180*p00+p00*p120*p180+p110*p00*p130+p170*p140*p00+p00*p160*p180+p150*p00*p190+p170*p200*p00+p00*p160*p200+p150*p00*p170+p190*p180*p00)-0.0135232*(p00*p100*p951+p90*p00*p961+p00*p100*p971+p90*p00*p981+p00*p120*p991+p110*p00*p1001+p00*p120*p1011+p110*p00*p1021+p00*p140*p1031+p130*p00*p1041+p00*p140*p1051+p130*p00*p1061+p00*p160*p1071+p150*p00*p1081+p00*p160*p1091+p150*p00*p1101+p00*p180*p1111+p170*p00*p1121+p00*p180*p1131+p170*p00*p1141+p00*p200*p1151+p190*p00*p1161+p00*p200*p1171+p190*p00*p1181)-0.00637036*(p00*p100*p950+p90*p00*p960+p00*p100*p970+p90*p00*p980+p00*p120*p990+p110*p00*p1000+p00*p120*p1010+p110*p00*p1020+p00*p140*p1030+p130*p00*p1040+p00*p140*p1050+p130*p00*p1060+p00*p160*p1070+p150*p00*p1080+p00*p160*p1090+p150*p00*p1100+p00*p180*p1110+p170*p00*p1120+p00*p180*p1130+p170*p00*p1140+p00*p200*p1150+p190*p00*p1160+p00*p200*p1170+p190*p00*p1180)-0.00476873*(p70*p00*p221+p1170*p210*p01+p70*p00*p241+p1090*p230*p01+p80*p00*p221+p1050*p210*p01+p80*p00*p241+p1010*p230*p01+p60*p00*p221+p1180*p210*p01+p60*p00*p251+p1130*p260*p01+p80*p00*p251+p970*p260*p01+p50*p00*p241+p1100*p230*p01+p50*p00*p251+p1140*p260*p01+p40*p00*p221+p1060*p210*p01+p40*p00*p261+p1110*p250*p01+p70*p00*p261+p950*p250*p01+p40*p00*p231+p1070*p240*p01+p60*p00*p231+p990*p240*p01+p30*p00*p241+p1020*p230*p01+p30*p00*p261+p1120*p250*p01+p20*p00*p251+p980*p260*p01+p20*p00*p231+p1080*p240*p01+p30*p00*p211+p1150*p220*p01+p50*p00*p211+p1030*p220*p01+p20*p00*p211+p1160*p220*p01+p10*p00*p261+p960*p250*p01+p10*p00*p231+p1000*p240*p01+p10*p00*p211+p1040*p220*p01)+0.00382591*(p01*p180*p90+p171*p00*p210+p101*p220*p00+p01*p170*p90+p181*p00*p230+p101*p240*p00+p01*p160*p110+p151*p00*p210+p121*p220*p00+p01*p110*p150+p121*p00*p250+p161*p260*p00+p01*p200*p130+p191*p00*p230+p141*p240*p00+p01*p130*p190+p141*p00*p250+p201*p260*p00+p01*p120*p150+p111*p00*p210+p161*p220*p00+p01*p100*p170+p91*p00*p210+p181*p220*p00+p01*p140*p190+p131*p00*p230+p201*p240*p00+p01*p200*p140+p191*p00*p250+p131*p260*p00+p01*p100*p180+p91*p00*p230+p171*p240*p00+p01*p160*p120+p151*p00*p250+p111*p260*p00)+0.059086*(p01*p181*p91+p171*p01*p211+p101*p221*p01+p01*p171*p91+p181*p01*p231+p101*p241*p01+p01*p161*p111+p151*p01*p211+p121*p221*p01+p01*p111*p151+p121*p01*p251+p161*p261*p01+p01*p201*p131+p191*p01*p231+p141*p241*p01+p01*p131*p191+p141*p01*p251+p201*p261*p01+p01*p121*p151+p111*p01*p211+p161*p221*p01+p01*p101*p171+p91*p01*p211+p181*p221*p01+p01*p141*p191+p131*p01*p231+p201*p241*p01+p01*p201*p141+p191*p01*p251+p131*p261*p01+p01*p101*p181+p91*p01*p231+p171*p241*p01+p01*p161*p121+p151*p01*p251+p111*p261*p01)+0.000178351*(p00*p180*p90+p170*p00*p210+p100*p220*p00+p00*p170*p90+p180*p00*p230+p100*p240*p00+p00*p160*p110+p150*p00*p210+p120*p220*p00+p00*p110*p150+p120*p00*p250+p160*p260*p00+p00*p200*p130+p190*p00*p230+p140*p240*p00+p00*p130*p190+p140*p00*p250+p200*p260*p00+p00*p120*p150+p110*p00*p210+p160*p220*p00+p00*p100*p170+p90*p00*p210+p180*p220*p00+p00*p140*p190+p130*p00*p230+p200*p240*p00+p00*p200*p140+p190*p00*p250+p130*p260*p00+p00*p100*p180+p90*p00*p230+p170*p240*p00+p00*p160*p120+p150*p00*p250+p110*p260*p00)-9.54342e-05*(p01*p100*p471+p91*p00*p381+p01*p90*p381+p101*p00*p471+p01*p100*p451+p91*p00*p341+p01*p90*p341+p101*p00*p451+p01*p100*p431+p91*p00*p321+p01*p90*p321+p101*p00*p431+p01*p100*p391+p91*p00*p301+p01*p90*p301+p101*p00*p391+p01*p120*p491+p111*p00*p401+p01*p110*p401+p121*p00*p491+p01*p120*p451+p111*p00*p361+p01*p110*p361+p121*p00*p451+p01*p120*p411+p111*p00*p321+p01*p110*p321+p121*p00*p411+p01*p120*p371+p111*p00*p281+p01*p110*p281+p121*p00*p371+p01*p140*p491+p131*p00*p441+p01*p130*p441+p141*p00*p491+p01*p140*p471+p131*p00*p421+p01*p130*p421+p141*p00*p471+p01*p140*p351+p131*p00*p301+p01*p130*p301+p141*p00*p351+p01*p140*p331+p131*p00*p281+p01*p130*p281+p141*p00*p331+p01*p160*p501+p151*p00*p461+p01*p150*p461+p161*p00*p501+p01*p160*p421+p151*p00*p381+p01*p150*p381+p161*p00*p421+p01*p160*p391+p151*p00*p351+p01*p150*p351+p161*p00*p391+p01*p160*p311+p151*p00*p271+p01*p150*p271+p161*p00*p311+p01*p180*p481+p171*p00*p461+p01*p170*p461+p181*p00*p481+p01*p180*p441+p171*p00*p401+p01*p170*p401+p181*p00*p441+p01*p180*p371+p171*p00*p331+p01*p170*p331+p181*p00*p371+p01*p180*p311+p171*p00*p291+p01*p170*p291+p181*p00*p311+p01*p200*p501+p191*p00*p481+p01*p190*p481+p201*p00*p501+p01*p200*p431+p191*p00*p411+p01*p190*p411+p201*p00*p431+p01*p200*p361+p191*p00*p341+p01*p190*p341+p201*p00*p361+p01*p200*p291+p191*p00*p271+p01*p190*p271+p201*p00*p291)+0.00382401*(p00*p121*p91+p110*p01*p511+p100*p521*p01+p00*p141*p91+p130*p01*p531+p100*p541*p01+p00*p161*p91+p150*p01*p551+p100*p561*p01+p00*p201*p91+p190*p01*p571+p100*p581*p01+p00*p101*p111+p90*p01*p511+p120*p521*p01+p00*p141*p111+p130*p01*p591+p120*p601*p01+p00*p181*p111+p170*p01*p611+p120*p621*p01+p00*p111*p191+p120*p01*p631+p200*p641*p01+p00*p101*p131+p90*p01*p531+p140*p541*p01+p00*p121*p131+p110*p01*p591+p140*p601*p01+p00*p171*p131+p180*p01*p651+p140*p661*p01+p00*p131*p151+p140*p01*p671+p160*p681*p01+p00*p101*p151+p90*p01*p551+p160*p561*p01+p00*p181*p151+p170*p01*p691+p160*p701*p01+p00*p201*p151+p190*p01*p711+p160*p721*p01+p00*p121*p171+p110*p01*p611+p180*p621*p01+p00*p161*p171+p150*p01*p691+p180*p701*p01+p00*p191*p171+p200*p01*p731+p180*p741*p01+p00*p101*p191+p90*p01*p571+p200*p581*p01+p00*p161*p191+p150*p01*p711+p200*p721*p01+p00*p201*p121+p190*p01*p631+p110*p641*p01+p00*p181*p201+p170*p01*p731+p190*p741*p01+p00*p141*p181+p130*p01*p651+p170*p661*p01+p00*p161*p141+p150*p01*p671+p130*p681*p01)-0.00299174*(p00*p120*p90+p110*p00*p510+p100*p520*p00+p00*p140*p90+p130*p00*p530+p100*p540*p00+p00*p160*p90+p150*p00*p550+p100*p560*p00+p00*p200*p90+p190*p00*p570+p100*p580*p00+p00*p100*p110+p90*p00*p510+p120*p520*p00+p00*p140*p110+p130*p00*p590+p120*p600*p00+p00*p180*p110+p170*p00*p610+p120*p620*p00+p00*p110*p190+p120*p00*p630+p200*p640*p00+p00*p100*p130+p90*p00*p530+p140*p540*p00+p00*p120*p130+p110*p00*p590+p140*p600*p00+p00*p170*p130+p180*p00*p650+p140*p660*p00+p00*p130*p150+p140*p00*p670+p160*p680*p00+p00*p100*p150+p90*p00*p550+p160*p560*p00+p00*p180*p150+p170*p00*p690+p160*p700*p00+p00*p200*p150+p190*p00*p710+p160*p720*p00+p00*p120*p170+p110*p00*p610+p180*p620*p00+p00*p160*p170+p150*p00*p690+p180*p700*p00+p00*p190*p170+p200*p00*p730+p180*p740*p00+p00*p100*p190+p90*p00*p570+p200*p580*p00+p00*p160*p190+p150*p00*p710+p200*p720*p00+p00*p200*p120+p190*p00*p630+p110*p640*p00+p00*p180*p200+p170*p00*p730+p190*p740*p00+p00*p140*p180+p130*p00*p650+p170*p660*p00+p00*p160*p140+p150*p00*p670+p130*p680*p00)-0.00270537*(p01*p90*p250+p101*p00*p600+p261*p590*p00+p01*p90*p260+p101*p00*p720+p251*p710*p00+p01*p110*p240+p121*p00*p540+p231*p530*p00+p01*p110*p230+p121*p00*p740+p241*p730*p00+p01*p130*p220+p141*p00*p520+p211*p510*p00+p01*p130*p210+p141*p00*p690+p221*p700*p00+p01*p150*p240+p161*p00*p580+p231*p570*p00+p01*p150*p230+p161*p00*p650+p241*p660*p00+p01*p170*p250+p181*p00*p630+p261*p640*p00+p01*p170*p260+p181*p00*p680+p251*p670*p00+p01*p190*p220+p201*p00*p560+p211*p550*p00+p01*p190*p210+p201*p00*p610+p221*p620*p00+p01*p200*p220+p191*p00*p620+p211*p610*p00+p01*p200*p210+p191*p00*p550+p221*p560*p00+p01*p180*p250+p171*p00*p670+p261*p680*p00+p01*p180*p260+p171*p00*p640+p251*p630*p00+p01*p160*p240+p151*p00*p660+p231*p650*p00+p01*p160*p230+p151*p00*p570+p241*p580*p00+p01*p140*p220+p131*p00*p700+p211*p690*p00+p01*p140*p210+p131*p00*p510+p221*p520*p00+p01*p120*p240+p111*p00*p730+p231*p740*p00+p01*p120*p230+p111*p00*p530+p241*p540*p00+p01*p100*p250+p91*p00*p710+p261*p720*p00+p01*p100*p260+p91*p00*p590+p251*p600*p00)+0.0179838*(p00*p90*p251+p100*p00*p601+p260*p590*p01+p00*p90*p261+p100*p00*p721+p250*p710*p01+p00*p110*p241+p120*p00*p541+p230*p530*p01+p00*p110*p231+p120*p00*p741+p240*p730*p01+p00*p130*p221+p140*p00*p521+p210*p510*p01+p00*p130*p211+p140*p00*p691+p220*p700*p01+p00*p150*p241+p160*p00*p581+p230*p570*p01+p00*p150*p231+p160*p00*p651+p240*p660*p01+p00*p170*p251+p180*p00*p631+p260*p640*p01+p00*p170*p261+p180*p00*p681+p250*p670*p01+p00*p190*p221+p200*p00*p561+p210*p550*p01+p00*p190*p211+p200*p00*p611+p220*p620*p01+p00*p200*p221+p190*p00*p621+p210*p610*p01+p00*p200*p211+p190*p00*p551+p220*p560*p01+p00*p180*p251+p170*p00*p671+p260*p680*p01+p00*p180*p261+p170*p00*p641+p250*p630*p01+p00*p160*p241+p150*p00*p661+p230*p650*p01+p00*p160*p231+p150*p00*p571+p240*p580*p01+p00*p140*p221+p130*p00*p701+p210*p690*p01+p00*p140*p211+p130*p00*p511+p220*p520*p01+p00*p120*p241+p110*p00*p731+p230*p740*p01+p00*p120*p231+p110*p00*p531+p240*p540*p01+p00*p100*p251+p90*p00*p711+p260*p720*p01+p00*p100*p261+p90*p00*p591+p250*p600*p01)+0.000756714*(p00*p90*p250+p100*p00*p600+p260*p590*p00+p00*p90*p260+p100*p00*p720+p250*p710*p00+p00*p110*p240+p120*p00*p540+p230*p530*p00+p00*p110*p230+p120*p00*p740+p240*p730*p00+p00*p130*p220+p140*p00*p520+p210*p510*p00+p00*p130*p210+p140*p00*p690+p220*p700*p00+p00*p150*p240+p160*p00*p580+p230*p570*p00+p00*p150*p230+p160*p00*p650+p240*p660*p00+p00*p170*p250+p180*p00*p630+p260*p640*p00+p00*p170*p260+p180*p00*p680+p250*p670*p00+p00*p190*p220+p200*p00*p560+p210*p550*p00+p00*p190*p210+p200*p00*p610+p220*p620*p00+p00*p200*p220+p190*p00*p620+p210*p610*p00+p00*p200*p210+p190*p00*p550+p220*p560*p00+p00*p180*p250+p170*p00*p670+p260*p680*p00+p00*p180*p260+p170*p00*p640+p250*p630*p00+p00*p160*p240+p150*p00*p660+p230*p650*p00+p00*p160*p230+p150*p00*p570+p240*p580*p00+p00*p140*p220+p130*p00*p700+p210*p690*p00+p00*p140*p210+p130*p00*p510+p220*p520*p00+p00*p120*p240+p110*p00*p730+p230*p740*p00+p00*p120*p230+p110*p00*p530+p240*p540*p00+p00*p100*p250+p90*p00*p710+p260*p720*p00+p00*p100*p260+p90*p00*p590+p250*p600*p00)+0.0234229*(p1030*p01*p700+p450*p691*p00+p990*p01*p730+p470*p741*p00+p1150*p01*p620+p390*p611*p00+p1070*p01*p660+p430*p651*p00+p950*p01*p710+p490*p721*p00+p1160*p01*p560+p370*p551*p00+p1110*p01*p670+p410*p681*p00+p1080*p01*p580+p330*p571*p00+p1120*p01*p630+p350*p641*p00+p1040*p01*p520+p310*p511*p00+p970*p01*p590+p500*p601*p00+p1130*p01*p640+p420*p631*p00+p1010*p01*p530+p480*p541*p00+p1090*p01*p570+p440*p581*p00+p1000*p01*p540+p290*p531*p00+p1140*p01*p680+p360*p671*p00+p960*p01*p600+p270*p591*p00+p1100*p01*p650+p340*p661*p00+p1050*p01*p510+p460*p521*p00+p1170*p01*p550+p400*p561*p00+p1180*p01*p610+p380*p621*p00+p980*p01*p720+p280*p711*p00+p1020*p01*p740+p300*p731*p00+p1060*p01*p690+p320*p701*p00)-0.00137209*(p71*p00*p520+p291*p510*p00+p71*p00*p540+p311*p530*p00+p81*p00*p560+p331*p550*p00+p81*p00*p580+p371*p570*p00+p61*p00*p520+p271*p510*p00+p61*p00*p600+p311*p590*p00+p81*p00*p620+p351*p610*p00+p81*p00*p630+p391*p640*p00+p51*p00*p540+p271*p530*p00+p51*p00*p600+p291*p590*p00+p81*p00*p660+p411*p650*p00+p81*p00*p670+p431*p680*p00+p41*p00*p560+p281*p550*p00+p71*p00*p700+p361*p690*p00+p41*p00*p720+p371*p710*p00+p71*p00*p680+p451*p670*p00+p41*p00*p620+p301*p610*p00+p61*p00*p700+p341*p690*p00+p41*p00*p740+p391*p730*p00+p61*p00*p650+p451*p660*p00+p31*p00*p580+p281*p570*p00+p31*p00*p720+p331*p710*p00+p71*p00*p730+p421*p740*p00+p71*p00*p640+p471*p630*p00+p21*p00*p630+p301*p640*p00+p21*p00*p740+p351*p730*p00+p61*p00*p710+p441*p720*p00+p61*p00*p570+p491*p580*p00+p31*p00*p660+p321*p650*p00+p51*p00*p730+p381*p740*p00+p31*p00*p690+p431*p700*p00+p51*p00*p610+p471*p620*p00+p21*p00*p670+p321*p680*p00+p51*p00*p710+p401*p720*p00+p21*p00*p690+p411*p700*p00+p51*p00*p550+p491*p560*p00+p11*p00*p680+p341*p670*p00+p11*p00*p650+p361*p660*p00+p41*p00*p590+p481*p600*p00+p41*p00*p530+p501*p540*p00+p11*p00*p640+p381*p630*p00+p11*p00*p610+p421*p620*p00+p31*p00*p590+p461*p600*p00+p31*p00*p510+p501*p520*p00+p11*p00*p570+p401*p580*p00+p11*p00*p550+p441*p560*p00+p21*p00*p530+p461*p540*p00+p21*p00*p510+p481*p520*p00)+0.0141868*(p01*p101*p620+p91*p01*p740+p611*p731*p00+p01*p101*p660+p91*p01*p690+p651*p701*p00+p01*p101*p700+p91*p01*p650+p691*p661*p00+p01*p101*p730+p91*p01*p610+p741*p621*p00+p01*p121*p560+p111*p01*p720+p551*p711*p00+p01*p121*p670+p111*p01*p690+p681*p701*p00+p01*p121*p700+p111*p01*p680+p691*p671*p00+p01*p121*p710+p111*p01*p550+p721*p561*p00+p01*p141*p580+p131*p01*p720+p571*p711*p00+p01*p141*p630+p131*p01*p740+p641*p731*p00+p01*p141*p730+p131*p01*p640+p741*p631*p00+p01*p141*p710+p131*p01*p570+p721*p581*p00+p01*p161*p520+p151*p01*p600+p511*p591*p00+p01*p161*p620+p151*p01*p630+p611*p641*p00+p01*p161*p640+p151*p01*p610+p631*p621*p00+p01*p161*p590+p151*p01*p510+p601*p521*p00+p01*p181*p520+p171*p01*p540+p511*p531*p00+p01*p181*p560+p171*p01*p580+p551*p571*p00+p01*p181*p570+p171*p01*p550+p581*p561*p00+p01*p181*p530+p171*p01*p510+p541*p521*p00+p01*p201*p540+p191*p01*p600+p531*p591*p00+p01*p201*p680+p191*p01*p650+p671*p661*p00+p01*p201*p660+p191*p01*p670+p651*p681*p00+p01*p201*p590+p191*p01*p530+p601*p541*p00)-0.00575271*(p00*p100*p621+p90*p00*p741+p610*p730*p01+p00*p100*p661+p90*p00*p691+p650*p700*p01+p00*p100*p701+p90*p00*p651+p690*p660*p01+p00*p100*p731+p90*p00*p611+p740*p620*p01+p00*p120*p561+p110*p00*p721+p550*p710*p01+p00*p120*p671+p110*p00*p691+p680*p700*p01+p00*p120*p701+p110*p00*p681+p690*p670*p01+p00*p120*p711+p110*p00*p551+p720*p560*p01+p00*p140*p581+p130*p00*p721+p570*p710*p01+p00*p140*p631+p130*p00*p741+p640*p730*p01+p00*p140*p731+p130*p00*p641+p740*p630*p01+p00*p140*p711+p130*p00*p571+p720*p580*p01+p00*p160*p521+p150*p00*p601+p510*p590*p01+p00*p160*p621+p150*p00*p631+p610*p640*p01+p00*p160*p641+p150*p00*p611+p630*p620*p01+p00*p160*p591+p150*p00*p511+p600*p520*p01+p00*p180*p521+p170*p00*p541+p510*p530*p01+p00*p180*p561+p170*p00*p581+p550*p570*p01+p00*p180*p571+p170*p00*p551+p580*p560*p01+p00*p180*p531+p170*p00*p511+p540*p520*p01+p00*p200*p541+p190*p00*p601+p530*p590*p01+p00*p200*p681+p190*p00*p651+p670*p660*p01+p00*p200*p661+p190*p00*p671+p650*p680*p01+p00*p200*p591+p190*p00*p531+p600*p540*p01)+0.0282786*(p01*p101*p621+p91*p01*p741+p611*p731*p01+p01*p101*p661+p91*p01*p691+p651*p701*p01+p01*p101*p701+p91*p01*p651+p691*p661*p01+p01*p101*p731+p91*p01*p611+p741*p621*p01+p01*p121*p561+p111*p01*p721+p551*p711*p01+p01*p121*p671+p111*p01*p691+p681*p701*p01+p01*p121*p701+p111*p01*p681+p691*p671*p01+p01*p121*p711+p111*p01*p551+p721*p561*p01+p01*p141*p581+p131*p01*p721+p571*p711*p01+p01*p141*p631+p131*p01*p741+p641*p731*p01+p01*p141*p731+p131*p01*p641+p741*p631*p01+p01*p141*p711+p131*p01*p571+p721*p581*p01+p01*p161*p521+p151*p01*p601+p511*p591*p01+p01*p161*p621+p151*p01*p631+p611*p641*p01+p01*p161*p641+p151*p01*p611+p631*p621*p01+p01*p161*p591+p151*p01*p511+p601*p521*p01+p01*p181*p521+p171*p01*p541+p511*p531*p01+p01*p181*p561+p171*p01*p581+p551*p571*p01+p01*p181*p571+p171*p01*p551+p581*p561*p01+p01*p181*p531+p171*p01*p511+p541*p521*p01+p01*p201*p541+p191*p01*p601+p531*p591*p01+p01*p201*p681+p191*p01*p651+p671*p661*p01+p01*p201*p661+p191*p01*p671+p651*p681*p01+p01*p201*p591+p191*p01*p531+p601*p541*p01)-0.00324915*(p00*p100*p620+p90*p00*p740+p610*p730*p00+p00*p100*p660+p90*p00*p690+p650*p700*p00+p00*p100*p700+p90*p00*p650+p690*p660*p00+p00*p100*p730+p90*p00*p610+p740*p620*p00+p00*p120*p560+p110*p00*p720+p550*p710*p00+p00*p120*p670+p110*p00*p690+p680*p700*p00+p00*p120*p700+p110*p00*p680+p690*p670*p00+p00*p120*p710+p110*p00*p550+p720*p560*p00+p00*p140*p580+p130*p00*p720+p570*p710*p00+p00*p140*p630+p130*p00*p740+p640*p730*p00+p00*p140*p730+p130*p00*p640+p740*p630*p00+p00*p140*p710+p130*p00*p570+p720*p580*p00+p00*p160*p520+p150*p00*p600+p510*p590*p00+p00*p160*p620+p150*p00*p630+p610*p640*p00+p00*p160*p640+p150*p00*p610+p630*p620*p00+p00*p160*p590+p150*p00*p510+p600*p520*p00+p00*p180*p520+p170*p00*p540+p510*p530*p00+p00*p180*p560+p170*p00*p580+p550*p570*p00+p00*p180*p570+p170*p00*p550+p580*p560*p00+p00*p180*p530+p170*p00*p510+p540*p520*p00+p00*p200*p540+p190*p00*p600+p530*p590*p00+p00*p200*p680+p190*p00*p650+p670*p660*p00+p00*p200*p660+p190*p00*p670+p650*p680*p00+p00*p200*p590+p190*p00*p530+p600*p540*p00)+0.0127299*(p00*p220*p541+p210*p00*p661+p530*p650*p01+p00*p240*p521+p230*p00*p621+p510*p610*p01+p00*p220*p581+p210*p00*p731+p570*p740*p01+p00*p240*p561+p230*p00*p701+p550*p690*p01+p00*p220*p601+p210*p00*p671+p590*p680*p01+p00*p260*p561+p250*p00*p521+p550*p510*p01+p00*p220*p631+p210*p00*p711+p640*p720*p01+p00*p260*p701+p250*p00*p621+p690*p610*p01+p00*p240*p601+p230*p00*p631+p590*p640*p01+p00*p260*p581+p250*p00*p541+p570*p530*p01+p00*p240*p671+p230*p00*p711+p680*p720*p01+p00*p260*p731+p250*p00*p661+p740*p650*p01+p00*p220*p721+p210*p00*p641+p710*p630*p01+p00*p220*p681+p210*p00*p591+p670*p600*p01+p00*p220*p741+p210*p00*p571+p730*p580*p01+p00*p220*p651+p210*p00*p531+p660*p540*p01+p00*p240*p721+p230*p00*p681+p710*p670*p01+p00*p240*p641+p230*p00*p591+p630*p600*p01+p00*p260*p651+p250*p00*p741+p660*p730*p01+p00*p260*p531+p250*p00*p571+p540*p580*p01+p00*p240*p691+p230*p00*p551+p700*p560*p01+p00*p240*p611+p230*p00*p511+p620*p520*p01+p00*p260*p611+p250*p00*p691+p620*p700*p01+p00*p260*p511+p250*p00*p551+p520*p560*p01)+0.018469*(p01*p220*p541+p211*p00*p661+p531*p650*p01+p01*p210*p661+p221*p00*p541+p651*p530*p01+p01*p240*p521+p231*p00*p621+p511*p610*p01+p01*p230*p621+p241*p00*p521+p611*p510*p01+p01*p220*p581+p211*p00*p731+p571*p740*p01+p01*p210*p731+p221*p00*p581+p741*p570*p01+p01*p240*p561+p231*p00*p701+p551*p690*p01+p01*p230*p701+p241*p00*p561+p691*p550*p01+p01*p220*p601+p211*p00*p671+p591*p680*p01+p01*p210*p671+p221*p00*p601+p681*p590*p01+p01*p260*p561+p251*p00*p521+p551*p510*p01+p01*p250*p521+p261*p00*p561+p511*p550*p01+p01*p220*p631+p211*p00*p711+p641*p720*p01+p01*p210*p711+p221*p00*p631+p721*p640*p01+p01*p260*p701+p251*p00*p621+p691*p610*p01+p01*p250*p621+p261*p00*p701+p611*p690*p01+p01*p240*p601+p231*p00*p631+p591*p640*p01+p01*p230*p631+p241*p00*p601+p641*p590*p01+p01*p260*p581+p251*p00*p541+p571*p530*p01+p01*p250*p541+p261*p00*p581+p531*p570*p01+p01*p240*p671+p231*p00*p711+p681*p720*p01+p01*p230*p711+p241*p00*p671+p721*p680*p01+p01*p260*p731+p251*p00*p661+p741*p650*p01+p01*p250*p661+p261*p00*p731+p651*p740*p01+p01*p220*p721+p211*p00*p641+p711*p630*p01+p01*p210*p641+p221*p00*p721+p631*p710*p01+p01*p220*p681+p211*p00*p591+p671*p600*p01+p01*p210*p591+p221*p00*p681+p601*p670*p01+p01*p220*p741+p211*p00*p571+p731*p580*p01+p01*p210*p571+p221*p00*p741+p581*p730*p01+p01*p220*p651+p211*p00*p531+p661*p540*p01+p01*p210*p531+p221*p00*p651+p541*p660*p01+p01*p240*p721+p231*p00*p681+p711*p670*p01+p01*p230*p681+p241*p00*p721+p671*p710*p01+p01*p240*p641+p231*p00*p591+p631*p600*p01+p01*p230*p591+p241*p00*p641+p601*p630*p01+p01*p260*p651+p251*p00*p741+p661*p730*p01+p01*p250*p741+p261*p00*p651+p731*p660*p01+p01*p260*p531+p251*p00*p571+p541*p580*p01+p01*p250*p571+p261*p00*p531+p581*p540*p01+p01*p240*p691+p231*p00*p551+p701*p560*p01+p01*p230*p551+p241*p00*p691+p561*p700*p01+p01*p240*p611+p231*p00*p511+p621*p520*p01+p01*p230*p511+p241*p00*p611+p521*p620*p01+p01*p260*p611+p251*p00*p691+p621*p700*p01+p01*p250*p691+p261*p00*p611+p701*p620*p01+p01*p260*p511+p251*p00*p551+p521*p560*p01+p01*p250*p551+p261*p00*p511+p561*p520*p01)+0.000193542*(p00*p220*p540+p210*p00*p660+p530*p650*p00+p00*p240*p520+p230*p00*p620+p510*p610*p00+p00*p220*p580+p210*p00*p730+p570*p740*p00+p00*p240*p560+p230*p00*p700+p550*p690*p00+p00*p220*p600+p210*p00*p670+p590*p680*p00+p00*p260*p560+p250*p00*p520+p550*p510*p00+p00*p220*p630+p210*p00*p710+p640*p720*p00+p00*p260*p700+p250*p00*p620+p690*p610*p00+p00*p240*p600+p230*p00*p630+p590*p640*p00+p00*p260*p580+p250*p00*p540+p570*p530*p00+p00*p240*p670+p230*p00*p710+p680*p720*p00+p00*p260*p730+p250*p00*p660+p740*p650*p00+p00*p220*p720+p210*p00*p640+p710*p630*p00+p00*p220*p680+p210*p00*p590+p670*p600*p00+p00*p220*p740+p210*p00*p570+p730*p580*p00+p00*p220*p650+p210*p00*p530+p660*p540*p00+p00*p240*p720+p230*p00*p680+p710*p670*p00+p00*p240*p640+p230*p00*p590+p630*p600*p00+p00*p260*p650+p250*p00*p740+p660*p730*p00+p00*p260*p530+p250*p00*p570+p540*p580*p00+p00*p240*p690+p230*p00*p550+p700*p560*p00+p00*p240*p610+p230*p00*p510+p620*p520*p00+p00*p260*p610+p250*p00*p690+p620*p700*p00+p00*p260*p510+p250*p00*p550+p520*p560*p00)-0.0144277*(p01*p540*p560+p531*p00*p680+p551*p670*p00+p01*p670*p550+p681*p00*p530+p561*p540*p00+p01*p680*p530+p671*p00*p550+p541*p560*p00+p01*p520*p580+p511*p00*p640+p571*p630*p00+p01*p630*p570+p641*p00*p510+p581*p520*p00+p01*p640*p510+p631*p00*p570+p521*p580*p00+p01*p660*p610+p651*p00*p590+p621*p600*p00+p01*p620*p600+p611*p00*p660+p591*p650*p00+p01*p590*p650+p601*p00*p620+p661*p610*p00+p01*p570*p510+p581*p00*p640+p521*p630*p00+p01*p580*p640+p571*p00*p510+p631*p520*p00+p01*p520*p630+p511*p00*p570+p641*p580*p00+p01*p620*p650+p611*p00*p590+p661*p600*p00+p01*p590*p610+p601*p00*p660+p621*p650*p00+p01*p660*p600+p651*p00*p620+p591*p610*p00+p01*p550*p530+p561*p00*p680+p541*p670*p00+p01*p560*p680+p551*p00*p530+p671*p540*p00+p01*p540*p670+p531*p00*p550+p681*p560*p00+p01*p730*p690+p741*p00*p710+p701*p720*p00+p01*p700*p720+p691*p00*p730+p711*p740*p00+p01*p710*p740+p721*p00*p700+p731*p690*p00+p01*p730*p720+p741*p00*p700+p711*p690*p00+p01*p700*p740+p691*p00*p710+p731*p720*p00+p01*p710*p690+p721*p00*p730+p701*p740*p00)-0.00362876*(p01*p541*p561+p531*p01*p681+p551*p671*p01+p01*p521*p581+p511*p01*p641+p571*p631*p01+p01*p661*p611+p651*p01*p591+p621*p601*p01+p01*p571*p511+p581*p01*p641+p521*p631*p01+p01*p621*p651+p611*p01*p591+p661*p601*p01+p01*p551*p531+p561*p01*p681+p541*p671*p01+p01*p731*p691+p741*p01*p711+p701*p721*p01+p01*p731*p721+p741*p01*p701+p711*p691*p01)+0.013012*(p1031*p00*p1190+p991*p00*p1200+p1151*p00*p1210+p1071*p00*p1220+p951*p00*p1230+p1161*p00*p1240+p1111*p00*p1250+p1081*p00*p1260+p1121*p00*p1270+p1041*p00*p1280+p971*p00*p1290+p1131*p00*p1300+p1011*p00*p1310+p1091*p00*p1320+p1001*p00*p1330+p1141*p00*p1340+p961*p00*p1350+p1101*p00*p1360+p1051*p00*p1370+p1171*p00*p1380+p1181*p00*p1390+p981*p00*p1400+p1021*p00*p1410+p1061*p00*p1420)+0.0155687*(p1030*p01*p1190+p990*p01*p1200+p1150*p01*p1210+p1070*p01*p1220+p950*p01*p1230+p1160*p01*p1240+p1110*p01*p1250+p1080*p01*p1260+p1120*p01*p1270+p1040*p01*p1280+p970*p01*p1290+p1130*p01*p1300+p1010*p01*p1310+p1090*p01*p1320+p1000*p01*p1330+p1140*p01*p1340+p960*p01*p1350+p1100*p01*p1360+p1050*p01*p1370+p1170*p01*p1380+p1180*p01*p1390+p980*p01*p1400+p1020*p01*p1410+p1060*p01*p1420)+0.0356897*(p1030*p00*p1191+p990*p00*p1201+p1150*p00*p1211+p1070*p00*p1221+p950*p00*p1231+p1160*p00*p1241+p1110*p00*p1251+p1080*p00*p1261+p1120*p00*p1271+p1040*p00*p1281+p970*p00*p1291+p1130*p00*p1301+p1010*p00*p1311+p1090*p00*p1321+p1000*p00*p1331+p1140*p00*p1341+p960*p00*p1351+p1100*p00*p1361+p1050*p00*p1371+p1170*p00*p1381+p1180*p00*p1391+p980*p00*p1401+p1020*p00*p1411+p1060*p00*p1421)+0.0236792*(p01*p90*p420+p101*p00*p1210+p01*p90*p360+p101*p00*p1220+p01*p90*p410+p101*p00*p1190+p01*p90*p350+p101*p00*p1200+p01*p110*p440+p121*p00*p1240+p01*p110*p340+p121*p00*p1250+p01*p110*p430+p121*p00*p1190+p01*p110*p330+p121*p00*p1230+p01*p130*p400+p141*p00*p1260+p01*p130*p380+p141*p00*p1270+p01*p130*p390+p141*p00*p1200+p01*p130*p370+p141*p00*p1230+p01*p150*p480+p161*p00*p1280+p01*p150*p470+p161*p00*p1210+p01*p150*p300+p161*p00*p1300+p01*p150*p290+p161*p00*p1290+p01*p170*p500+p181*p00*p1280+p01*p170*p490+p181*p00*p1240+p01*p170*p280+p181*p00*p1320+p01*p170*p270+p181*p00*p1310+p01*p190*p460+p201*p00*p1330+p01*p190*p320+p201*p00*p1340+p01*p190*p450+p201*p00*p1220+p01*p190*p310+p201*p00*p1290+p01*p200*p460+p191*p00*p1350+p01*p200*p320+p191*p00*p1360+p01*p200*p450+p191*p00*p1250+p01*p200*p310+p191*p00*p1310+p01*p180*p500+p171*p00*p1330+p01*p180*p490+p171*p00*p1260+p01*p180*p280+p171*p00*p1380+p01*p180*p270+p171*p00*p1370+p01*p160*p480+p151*p00*p1350+p01*p160*p470+p151*p00*p1270+p01*p160*p300+p151*p00*p1390+p01*p160*p290+p151*p00*p1370+p01*p140*p400+p131*p00*p1400+p01*p140*p380+p131*p00*p1410+p01*p140*p390+p131*p00*p1300+p01*p140*p370+p131*p00*p1320+p01*p120*p440+p111*p00*p1400+p01*p120*p340+p111*p00*p1420+p01*p120*p430+p111*p00*p1340+p01*p120*p330+p111*p00*p1380+p01*p100*p420+p91*p00*p1410+p01*p100*p360+p91*p00*p1420+p01*p100*p410+p91*p00*p1360+p01*p100*p350+p91*p00*p1390)+0.0034611*(p00*p91*p420+p100*p01*p1210+p00*p91*p360+p100*p01*p1220+p00*p91*p410+p100*p01*p1190+p00*p91*p350+p100*p01*p1200+p00*p111*p440+p120*p01*p1240+p00*p111*p340+p120*p01*p1250+p00*p111*p430+p120*p01*p1190+p00*p111*p330+p120*p01*p1230+p00*p131*p400+p140*p01*p1260+p00*p131*p380+p140*p01*p1270+p00*p131*p390+p140*p01*p1200+p00*p131*p370+p140*p01*p1230+p00*p151*p480+p160*p01*p1280+p00*p151*p470+p160*p01*p1210+p00*p151*p300+p160*p01*p1300+p00*p151*p290+p160*p01*p1290+p00*p171*p500+p180*p01*p1280+p00*p171*p490+p180*p01*p1240+p00*p171*p280+p180*p01*p1320+p00*p171*p270+p180*p01*p1310+p00*p191*p460+p200*p01*p1330+p00*p191*p320+p200*p01*p1340+p00*p191*p450+p200*p01*p1220+p00*p191*p310+p200*p01*p1290+p00*p201*p460+p190*p01*p1350+p00*p201*p320+p190*p01*p1360+p00*p201*p450+p190*p01*p1250+p00*p201*p310+p190*p01*p1310+p00*p181*p500+p170*p01*p1330+p00*p181*p490+p170*p01*p1260+p00*p181*p280+p170*p01*p1380+p00*p181*p270+p170*p01*p1370+p00*p161*p480+p150*p01*p1350+p00*p161*p470+p150*p01*p1270+p00*p161*p300+p150*p01*p1390+p00*p161*p290+p150*p01*p1370+p00*p141*p400+p130*p01*p1400+p00*p141*p380+p130*p01*p1410+p00*p141*p390+p130*p01*p1300+p00*p141*p370+p130*p01*p1320+p00*p121*p440+p110*p01*p1400+p00*p121*p340+p110*p01*p1420+p00*p121*p430+p110*p01*p1340+p00*p121*p330+p110*p01*p1380+p00*p101*p420+p90*p01*p1410+p00*p101*p360+p90*p01*p1420+p00*p101*p410+p90*p01*p1360+p00*p101*p350+p90*p01*p1390)-0.000686215*(p01*p240*p210+p231*p00*p830+p221*p840*p00+p01*p220*p230+p211*p00*p830+p241*p840*p00+p01*p250*p210+p261*p00*p850+p221*p860*p00+p01*p220*p260+p211*p00*p850+p251*p860*p00+p01*p250*p230+p261*p00*p870+p241*p880*p00+p01*p240*p260+p231*p00*p870+p251*p880*p00+p01*p260*p210+p251*p00*p890+p221*p900*p00+p01*p220*p250+p211*p00*p890+p261*p900*p00+p01*p230*p210+p241*p00*p910+p221*p920*p00+p01*p220*p240+p211*p00*p910+p231*p920*p00+p01*p260*p230+p251*p00*p930+p241*p940*p00+p01*p240*p250+p231*p00*p930+p261*p940*p00)-0.0118789*(p00*p241*p210+p230*p01*p830+p220*p841*p00+p00*p211*p240+p220*p01*p840+p230*p831*p00+p00*p221*p230+p210*p01*p830+p240*p841*p00+p00*p231*p220+p240*p01*p840+p210*p831*p00+p00*p251*p210+p260*p01*p850+p220*p861*p00+p00*p211*p250+p220*p01*p860+p260*p851*p00+p00*p221*p260+p210*p01*p850+p250*p861*p00+p00*p261*p220+p250*p01*p860+p210*p851*p00+p00*p251*p230+p260*p01*p870+p240*p881*p00+p00*p231*p250+p240*p01*p880+p260*p871*p00+p00*p241*p260+p230*p01*p870+p250*p881*p00+p00*p261*p240+p250*p01*p880+p230*p871*p00+p00*p261*p210+p250*p01*p890+p220*p901*p00+p00*p211*p260+p220*p01*p900+p250*p891*p00+p00*p221*p250+p210*p01*p890+p260*p901*p00+p00*p251*p220+p260*p01*p900+p210*p891*p00+p00*p231*p210+p240*p01*p910+p220*p921*p00+p00*p211*p230+p220*p01*p920+p240*p911*p00+p00*p221*p240+p210*p01*p910+p230*p921*p00+p00*p241*p220+p230*p01*p920+p210*p911*p00+p00*p261*p230+p250*p01*p930+p240*p941*p00+p00*p231*p260+p240*p01*p940+p250*p931*p00+p00*p241*p250+p230*p01*p930+p260*p941*p00+p00*p251*p240+p260*p01*p940+p230*p931*p00)+0.0159041*(p01*p241*p210+p231*p01*p830+p221*p841*p00+p01*p211*p240+p221*p01*p840+p231*p831*p00+p01*p221*p230+p211*p01*p830+p241*p841*p00+p01*p231*p220+p241*p01*p840+p211*p831*p00+p01*p251*p210+p261*p01*p850+p221*p861*p00+p01*p211*p250+p221*p01*p860+p261*p851*p00+p01*p221*p260+p211*p01*p850+p251*p861*p00+p01*p261*p220+p251*p01*p860+p211*p851*p00+p01*p251*p230+p261*p01*p870+p241*p881*p00+p01*p231*p250+p241*p01*p880+p261*p871*p00+p01*p241*p260+p231*p01*p870+p251*p881*p00+p01*p261*p240+p251*p01*p880+p231*p871*p00+p01*p261*p210+p251*p01*p890+p221*p901*p00+p01*p211*p260+p221*p01*p900+p251*p891*p00+p01*p221*p250+p211*p01*p890+p261*p901*p00+p01*p251*p220+p261*p01*p900+p211*p891*p00+p01*p231*p210+p241*p01*p910+p221*p921*p00+p01*p211*p230+p221*p01*p920+p241*p911*p00+p01*p221*p240+p211*p01*p910+p231*p921*p00+p01*p241*p220+p231*p01*p920+p211*p911*p00+p01*p261*p230+p251*p01*p930+p241*p941*p00+p01*p231*p260+p241*p01*p940+p251*p931*p00+p01*p241*p250+p231*p01*p930+p261*p941*p00+p01*p251*p240+p261*p01*p940+p231*p931*p00)-0.00970363*(p01*p241*p211+p231*p01*p831+p221*p841*p01+p01*p221*p231+p211*p01*p831+p241*p841*p01+p01*p251*p211+p261*p01*p851+p221*p861*p01+p01*p221*p261+p211*p01*p851+p251*p861*p01+p01*p251*p231+p261*p01*p871+p241*p881*p01+p01*p241*p261+p231*p01*p871+p251*p881*p01+p01*p261*p211+p251*p01*p891+p221*p901*p01+p01*p221*p251+p211*p01*p891+p261*p901*p01+p01*p231*p211+p241*p01*p911+p221*p921*p01+p01*p221*p241+p211*p01*p911+p231*p921*p01+p01*p261*p231+p251*p01*p931+p241*p941*p01+p01*p241*p251+p231*p01*p931+p261*p941*p01)-0.00220969*(p00*p240*p210+p230*p00*p830+p220*p840*p00+p00*p220*p230+p210*p00*p830+p240*p840*p00+p00*p250*p210+p260*p00*p850+p220*p860*p00+p00*p220*p260+p210*p00*p850+p250*p860*p00+p00*p250*p230+p260*p00*p870+p240*p880*p00+p00*p240*p260+p230*p00*p870+p250*p880*p00+p00*p260*p210+p250*p00*p890+p220*p900*p00+p00*p220*p250+p210*p00*p890+p260*p900*p00+p00*p230*p210+p240*p00*p910+p220*p920*p00+p00*p220*p240+p210*p00*p910+p230*p920*p00+p00*p260*p230+p250*p00*p930+p240*p940*p00+p00*p240*p250+p230*p00*p930+p260*p940*p00)-0.0267446*(p01*p101*p90+p91*p01*p830+p101*p841*p00+p01*p91*p100+p101*p01*p840+p91*p831*p00+p01*p121*p110+p111*p01*p850+p121*p861*p00+p01*p111*p120+p121*p01*p860+p111*p851*p00+p01*p141*p130+p131*p01*p870+p141*p881*p00+p01*p131*p140+p141*p01*p880+p131*p871*p00+p01*p161*p150+p151*p01*p890+p161*p901*p00+p01*p151*p160+p161*p01*p900+p151*p891*p00+p01*p181*p170+p171*p01*p910+p181*p921*p00+p01*p171*p180+p181*p01*p920+p171*p911*p00+p01*p201*p190+p191*p01*p930+p201*p941*p00+p01*p191*p200+p201*p01*p940+p191*p931*p00)+0.0727491*(p00*p101*p91+p90*p01*p831+p100*p841*p01+p00*p121*p111+p110*p01*p851+p120*p861*p01+p00*p141*p131+p130*p01*p871+p140*p881*p01+p00*p161*p151+p150*p01*p891+p160*p901*p01+p00*p181*p171+p170*p01*p911+p180*p921*p01+p00*p201*p191+p190*p01*p931+p200*p941*p01)-0.00283388*(p00*p100*p90+p90*p00*p830+p100*p840*p00+p00*p120*p110+p110*p00*p850+p120*p860*p00+p00*p140*p130+p130*p00*p870+p140*p880*p00+p00*p160*p150+p150*p00*p890+p160*p900*p00+p00*p180*p170+p170*p00*p910+p180*p920*p00+p00*p200*p190+p190*p00*p930+p200*p940*p00)+0.0120949*(p450*p01*p890+p490*p901*p00+p490*p01*p900+p450*p891*p00+p470*p01*p930+p490*p941*p00+p490*p01*p940+p470*p931*p00+p390*p01*p850+p500*p861*p00+p500*p01*p860+p390*p851*p00+p430*p01*p870+p500*p881*p00+p500*p01*p880+p430*p871*p00+p450*p01*p910+p470*p921*p00+p470*p01*p920+p450*p911*p00+p370*p01*p830+p480*p841*p00+p480*p01*p840+p370*p831*p00+p410*p01*p870+p480*p881*p00+p480*p01*p880+p410*p871*p00+p330*p01*p830+p460*p841*p00+p460*p01*p840+p330*p831*p00+p350*p01*p850+p460*p861*p00+p460*p01*p860+p350*p851*p00+p390*p01*p910+p430*p921*p00+p430*p01*p920+p390*p911*p00+p310*p01*p830+p440*p841*p00+p440*p01*p840+p310*p831*p00+p420*p01*p930+p440*p941*p00+p440*p01*p940+p420*p931*p00+p370*p01*p890+p410*p901*p00+p410*p01*p900+p370*p891*p00+p310*p01*p850+p420*p861*p00+p420*p01*p860+p310*p851*p00+p290*p01*p830+p400*p841*p00+p400*p01*p840+p290*p831*p00+p360*p01*p890+p400*p901*p00+p400*p01*p900+p360*p891*p00+p270*p01*p850+p380*p861*p00+p380*p01*p860+p270*p851*p00+p340*p01*p910+p380*p921*p00+p380*p01*p920+p340*p911*p00+p330*p01*p930+p350*p941*p00+p350*p01*p940+p330*p931*p00+p290*p01*p870+p360*p881*p00+p360*p01*p880+p290*p871*p00+p270*p01*p870+p340*p881*p00+p340*p01*p880+p270*p871*p00+p280*p01*p890+p320*p901*p00+p320*p01*p900+p280*p891*p00+p300*p01*p910+p320*p921*p00+p320*p01*p920+p300*p911*p00+p280*p01*p930+p300*p941*p00+p300*p01*p940+p280*p931*p00)+0.00357958*(p01*p90*p630+p101*p00*p860+p641*p850*p00+p01*p90*p670+p101*p00*p880+p681*p870*p00+p01*p90*p680+p101*p00*p900+p671*p890*p00+p01*p90*p640+p101*p00*p940+p631*p930*p00+p01*p110*p580+p121*p00*p840+p571*p830*p00+p01*p110*p660+p121*p00*p880+p651*p870*p00+p01*p110*p650+p121*p00*p920+p661*p910*p00+p01*p110*p570+p121*p00*p930+p581*p940*p00+p01*p130*p560+p141*p00*p840+p551*p830*p00+p01*p130*p620+p141*p00*p860+p611*p850*p00+p01*p130*p610+p141*p00*p910+p621*p920*p00+p01*p130*p550+p141*p00*p890+p561*p900*p00+p01*p150*p540+p161*p00*p840+p531*p830*p00+p01*p150*p740+p161*p00*p920+p731*p910*p00+p01*p150*p730+p161*p00*p940+p741*p930*p00+p01*p150*p530+p161*p00*p870+p541*p880*p00+p01*p170*p600+p181*p00*p860+p591*p850*p00+p01*p170*p720+p181*p00*p900+p711*p890*p00+p01*p170*p710+p181*p00*p930+p721*p940*p00+p01*p170*p590+p181*p00*p870+p601*p880*p00+p01*p190*p520+p201*p00*p840+p511*p830*p00+p01*p190*p700+p201*p00*p900+p691*p890*p00+p01*p190*p690+p201*p00*p910+p701*p920*p00+p01*p190*p510+p201*p00*p850+p521*p860*p00+p01*p200*p520+p191*p00*p860+p511*p850*p00+p01*p200*p700+p191*p00*p920+p691*p910*p00+p01*p200*p690+p191*p00*p890+p701*p900*p00+p01*p200*p510+p191*p00*p830+p521*p840*p00+p01*p180*p600+p171*p00*p880+p591*p870*p00+p01*p180*p720+p171*p00*p940+p711*p930*p00+p01*p180*p710+p171*p00*p890+p721*p900*p00+p01*p180*p590+p171*p00*p850+p601*p860*p00+p01*p160*p540+p151*p00*p880+p531*p870*p00+p01*p160*p740+p151*p00*p930+p731*p940*p00+p01*p160*p730+p151*p00*p910+p741*p920*p00+p01*p160*p530+p151*p00*p830+p541*p840*p00+p01*p140*p560+p131*p00*p900+p551*p890*p00+p01*p140*p620+p131*p00*p920+p611*p910*p00+p01*p140*p610+p131*p00*p850+p621*p860*p00+p01*p140*p550+p131*p00*p830+p561*p840*p00+p01*p120*p580+p111*p00*p940+p571*p930*p00+p01*p120*p660+p111*p00*p910+p651*p920*p00+p01*p120*p650+p111*p00*p870+p661*p880*p00+p01*p120*p570+p111*p00*p830+p581*p840*p00+p01*p100*p630+p91*p00*p930+p641*p940*p00+p01*p100*p670+p91*p00*p890+p681*p900*p00+p01*p100*p680+p91*p00*p870+p671*p880*p00+p01*p100*p640+p91*p00*p850+p631*p860*p00)+0.00295353*(p00*p91*p630+p100*p01*p860+p640*p851*p00+p00*p91*p670+p100*p01*p880+p680*p871*p00+p00*p91*p680+p100*p01*p900+p670*p891*p00+p00*p91*p640+p100*p01*p940+p630*p931*p00+p00*p111*p580+p120*p01*p840+p570*p831*p00+p00*p111*p660+p120*p01*p880+p650*p871*p00+p00*p111*p650+p120*p01*p920+p660*p911*p00+p00*p111*p570+p120*p01*p930+p580*p941*p00+p00*p131*p560+p140*p01*p840+p550*p831*p00+p00*p131*p620+p140*p01*p860+p610*p851*p00+p00*p131*p610+p140*p01*p910+p620*p921*p00+p00*p131*p550+p140*p01*p890+p560*p901*p00+p00*p151*p540+p160*p01*p840+p530*p831*p00+p00*p151*p740+p160*p01*p920+p730*p911*p00+p00*p151*p730+p160*p01*p940+p740*p931*p00+p00*p151*p530+p160*p01*p870+p540*p881*p00+p00*p171*p600+p180*p01*p860+p590*p851*p00+p00*p171*p720+p180*p01*p900+p710*p891*p00+p00*p171*p710+p180*p01*p930+p720*p941*p00+p00*p171*p590+p180*p01*p870+p600*p881*p00+p00*p191*p520+p200*p01*p840+p510*p831*p00+p00*p191*p700+p200*p01*p900+p690*p891*p00+p00*p191*p690+p200*p01*p910+p700*p921*p00+p00*p191*p510+p200*p01*p850+p520*p861*p00+p00*p201*p520+p190*p01*p860+p510*p851*p00+p00*p201*p700+p190*p01*p920+p690*p911*p00+p00*p201*p690+p190*p01*p890+p700*p901*p00+p00*p201*p510+p190*p01*p830+p520*p841*p00+p00*p181*p600+p170*p01*p880+p590*p871*p00+p00*p181*p720+p170*p01*p940+p710*p931*p00+p00*p181*p710+p170*p01*p890+p720*p901*p00+p00*p181*p590+p170*p01*p850+p600*p861*p00+p00*p161*p540+p150*p01*p880+p530*p871*p00+p00*p161*p740+p150*p01*p930+p730*p941*p00+p00*p161*p730+p150*p01*p910+p740*p921*p00+p00*p161*p530+p150*p01*p830+p540*p841*p00+p00*p141*p560+p130*p01*p900+p550*p891*p00+p00*p141*p620+p130*p01*p920+p610*p911*p00+p00*p141*p610+p130*p01*p850+p620*p861*p00+p00*p141*p550+p130*p01*p830+p560*p841*p00+p00*p121*p580+p110*p01*p940+p570*p931*p00+p00*p121*p660+p110*p01*p910+p650*p921*p00+p00*p121*p650+p110*p01*p870+p660*p881*p00+p00*p121*p570+p110*p01*p830+p580*p841*p00+p00*p101*p630+p90*p01*p930+p640*p941*p00+p00*p101*p670+p90*p01*p890+p680*p901*p00+p00*p101*p680+p90*p01*p870+p670*p881*p00+p00*p101*p640+p90*p01*p850+p630*p861*p00)-0.00486714*(p00*p90*p631+p100*p00*p861+p640*p850*p01+p00*p90*p671+p100*p00*p881+p680*p870*p01+p00*p90*p681+p100*p00*p901+p670*p890*p01+p00*p90*p641+p100*p00*p941+p630*p930*p01+p00*p110*p581+p120*p00*p841+p570*p830*p01+p00*p110*p661+p120*p00*p881+p650*p870*p01+p00*p110*p651+p120*p00*p921+p660*p910*p01+p00*p110*p571+p120*p00*p931+p580*p940*p01+p00*p130*p561+p140*p00*p841+p550*p830*p01+p00*p130*p621+p140*p00*p861+p610*p850*p01+p00*p130*p611+p140*p00*p911+p620*p920*p01+p00*p130*p551+p140*p00*p891+p560*p900*p01+p00*p150*p541+p160*p00*p841+p530*p830*p01+p00*p150*p741+p160*p00*p921+p730*p910*p01+p00*p150*p731+p160*p00*p941+p740*p930*p01+p00*p150*p531+p160*p00*p871+p540*p880*p01+p00*p170*p601+p180*p00*p861+p590*p850*p01+p00*p170*p721+p180*p00*p901+p710*p890*p01+p00*p170*p711+p180*p00*p931+p720*p940*p01+p00*p170*p591+p180*p00*p871+p600*p880*p01+p00*p190*p521+p200*p00*p841+p510*p830*p01+p00*p190*p701+p200*p00*p901+p690*p890*p01+p00*p190*p691+p200*p00*p911+p700*p920*p01+p00*p190*p511+p200*p00*p851+p520*p860*p01+p00*p200*p521+p190*p00*p861+p510*p850*p01+p00*p200*p701+p190*p00*p921+p690*p910*p01+p00*p200*p691+p190*p00*p891+p700*p900*p01+p00*p200*p511+p190*p00*p831+p520*p840*p01+p00*p180*p601+p170*p00*p881+p590*p870*p01+p00*p180*p721+p170*p00*p941+p710*p930*p01+p00*p180*p711+p170*p00*p891+p720*p900*p01+p00*p180*p591+p170*p00*p851+p600*p860*p01+p00*p160*p541+p150*p00*p881+p530*p870*p01+p00*p160*p741+p150*p00*p931+p730*p940*p01+p00*p160*p731+p150*p00*p911+p740*p920*p01+p00*p160*p531+p150*p00*p831+p540*p840*p01+p00*p140*p561+p130*p00*p901+p550*p890*p01+p00*p140*p621+p130*p00*p921+p610*p910*p01+p00*p140*p611+p130*p00*p851+p620*p860*p01+p00*p140*p551+p130*p00*p831+p560*p840*p01+p00*p120*p581+p110*p00*p941+p570*p930*p01+p00*p120*p661+p110*p00*p911+p650*p920*p01+p00*p120*p651+p110*p00*p871+p660*p880*p01+p00*p120*p571+p110*p00*p831+p580*p840*p01+p00*p100*p631+p90*p00*p931+p640*p940*p01+p00*p100*p671+p90*p00*p891+p680*p900*p01+p00*p100*p681+p90*p00*p871+p670*p880*p01+p00*p100*p641+p90*p00*p851+p630*p860*p01)-0.0104047*(p01*p91*p631+p101*p01*p861+p641*p851*p01+p01*p91*p671+p101*p01*p881+p681*p871*p01+p01*p91*p681+p101*p01*p901+p671*p891*p01+p01*p91*p641+p101*p01*p941+p631*p931*p01+p01*p111*p581+p121*p01*p841+p571*p831*p01+p01*p111*p661+p121*p01*p881+p651*p871*p01+p01*p111*p651+p121*p01*p921+p661*p911*p01+p01*p111*p571+p121*p01*p931+p581*p941*p01+p01*p131*p561+p141*p01*p841+p551*p831*p01+p01*p131*p621+p141*p01*p861+p611*p851*p01+p01*p131*p611+p141*p01*p911+p621*p921*p01+p01*p131*p551+p141*p01*p891+p561*p901*p01+p01*p151*p541+p161*p01*p841+p531*p831*p01+p01*p151*p741+p161*p01*p921+p731*p911*p01+p01*p151*p731+p161*p01*p941+p741*p931*p01+p01*p151*p531+p161*p01*p871+p541*p881*p01+p01*p171*p601+p181*p01*p861+p591*p851*p01+p01*p171*p721+p181*p01*p901+p711*p891*p01+p01*p171*p711+p181*p01*p931+p721*p941*p01+p01*p171*p591+p181*p01*p871+p601*p881*p01+p01*p191*p521+p201*p01*p841+p511*p831*p01+p01*p191*p701+p201*p01*p901+p691*p891*p01+p01*p191*p691+p201*p01*p911+p701*p921*p01+p01*p191*p511+p201*p01*p851+p521*p861*p01+p01*p201*p521+p191*p01*p861+p511*p851*p01+p01*p201*p701+p191*p01*p921+p691*p911*p01+p01*p201*p691+p191*p01*p891+p701*p901*p01+p01*p201*p511+p191*p01*p831+p521*p841*p01+p01*p181*p601+p171*p01*p881+p591*p871*p01+p01*p181*p721+p171*p01*p941+p711*p931*p01+p01*p181*p711+p171*p01*p891+p721*p901*p01+p01*p181*p591+p171*p01*p851+p601*p861*p01+p01*p161*p541+p151*p01*p881+p531*p871*p01+p01*p161*p741+p151*p01*p931+p731*p941*p01+p01*p161*p731+p151*p01*p911+p741*p921*p01+p01*p161*p531+p151*p01*p831+p541*p841*p01+p01*p141*p561+p131*p01*p901+p551*p891*p01+p01*p141*p621+p131*p01*p921+p611*p911*p01+p01*p141*p611+p131*p01*p851+p621*p861*p01+p01*p141*p551+p131*p01*p831+p561*p841*p01+p01*p121*p581+p111*p01*p941+p571*p931*p01+p01*p121*p661+p111*p01*p911+p651*p921*p01+p01*p121*p651+p111*p01*p871+p661*p881*p01+p01*p121*p571+p111*p01*p831+p581*p841*p01+p01*p101*p631+p91*p01*p931+p641*p941*p01+p01*p101*p671+p91*p01*p891+p681*p901*p01+p01*p101*p681+p91*p01*p871+p671*p881*p01+p01*p101*p641+p91*p01*p851+p631*p861*p01)+0.00063407*(p00*p90*p630+p100*p00*p860+p640*p850*p00+p00*p90*p670+p100*p00*p880+p680*p870*p00+p00*p90*p680+p100*p00*p900+p670*p890*p00+p00*p90*p640+p100*p00*p940+p630*p930*p00+p00*p110*p580+p120*p00*p840+p570*p830*p00+p00*p110*p660+p120*p00*p880+p650*p870*p00+p00*p110*p650+p120*p00*p920+p660*p910*p00+p00*p110*p570+p120*p00*p930+p580*p940*p00+p00*p130*p560+p140*p00*p840+p550*p830*p00+p00*p130*p620+p140*p00*p860+p610*p850*p00+p00*p130*p610+p140*p00*p910+p620*p920*p00+p00*p130*p550+p140*p00*p890+p560*p900*p00+p00*p150*p540+p160*p00*p840+p530*p830*p00+p00*p150*p740+p160*p00*p920+p730*p910*p00+p00*p150*p730+p160*p00*p940+p740*p930*p00+p00*p150*p530+p160*p00*p870+p540*p880*p00+p00*p170*p600+p180*p00*p860+p590*p850*p00+p00*p170*p720+p180*p00*p900+p710*p890*p00+p00*p170*p710+p180*p00*p930+p720*p940*p00+p00*p170*p590+p180*p00*p870+p600*p880*p00+p00*p190*p520+p200*p00*p840+p510*p830*p00+p00*p190*p700+p200*p00*p900+p690*p890*p00+p00*p190*p690+p200*p00*p910+p700*p920*p00+p00*p190*p510+p200*p00*p850+p520*p860*p00+p00*p200*p520+p190*p00*p860+p510*p850*p00+p00*p200*p700+p190*p00*p920+p690*p910*p00+p00*p200*p690+p190*p00*p890+p700*p900*p00+p00*p200*p510+p190*p00*p830+p520*p840*p00+p00*p180*p600+p170*p00*p880+p590*p870*p00+p00*p180*p720+p170*p00*p940+p710*p930*p00+p00*p180*p710+p170*p00*p890+p720*p900*p00+p00*p180*p590+p170*p00*p850+p600*p860*p00+p00*p160*p540+p150*p00*p880+p530*p870*p00+p00*p160*p740+p150*p00*p930+p730*p940*p00+p00*p160*p730+p150*p00*p910+p740*p920*p00+p00*p160*p530+p150*p00*p830+p540*p840*p00+p00*p140*p560+p130*p00*p900+p550*p890*p00+p00*p140*p620+p130*p00*p920+p610*p910*p00+p00*p140*p610+p130*p00*p850+p620*p860*p00+p00*p140*p550+p130*p00*p830+p560*p840*p00+p00*p120*p580+p110*p00*p940+p570*p930*p00+p00*p120*p660+p110*p00*p910+p650*p920*p00+p00*p120*p650+p110*p00*p870+p660*p880*p00+p00*p120*p570+p110*p00*p830+p580*p840*p00+p00*p100*p630+p90*p00*p930+p640*p940*p00+p00*p100*p670+p90*p00*p890+p680*p900*p00+p00*p100*p680+p90*p00*p870+p670*p880*p00+p00*p100*p640+p90*p00*p850+p630*p860*p00)+0.0114912*(p01*p610*p940+p621*p00*p560+p931*p550*p00+p01*p550*p930+p561*p00*p620+p941*p610*p00+p01*p650*p900+p661*p00*p580+p891*p570*p00+p01*p570*p890+p581*p00*p660+p901*p650*p00+p01*p690*p880+p701*p00*p520+p871*p510*p00+p01*p510*p870+p521*p00*p700+p881*p690*p00+p01*p740*p860+p731*p00*p540+p851*p530*p00+p01*p530*p850+p541*p00*p730+p861*p740*p00+p01*p680*p920+p671*p00*p630+p911*p640*p00+p01*p640*p910+p631*p00*p670+p921*p680*p00+p01*p720*p840+p711*p00*p600+p831*p590*p00+p01*p590*p830+p601*p00*p710+p841*p720*p00+p01*p630*p920+p641*p00*p680+p911*p670*p00+p01*p670*p910+p681*p00*p640+p921*p630*p00+p01*p600*p840+p591*p00*p720+p831*p710*p00+p01*p710*p830+p721*p00*p590+p841*p600*p00+p01*p580*p900+p571*p00*p650+p891*p660*p00+p01*p660*p890+p651*p00*p570+p901*p580*p00+p01*p540*p860+p531*p00*p740+p851*p730*p00+p01*p730*p850+p741*p00*p530+p861*p540*p00+p01*p560*p940+p551*p00*p610+p931*p620*p00+p01*p620*p930+p611*p00*p550+p941*p560*p00+p01*p520*p880+p511*p00*p690+p871*p700*p00+p01*p700*p870+p691*p00*p510+p881*p520*p00)-0.00463662*(p00*p611*p940+p620*p01*p560+p930*p551*p00+p00*p651*p900+p660*p01*p580+p890*p571*p00+p00*p691*p880+p700*p01*p520+p870*p511*p00+p00*p741*p860+p730*p01*p540+p850*p531*p00+p00*p681*p920+p670*p01*p630+p910*p641*p00+p00*p721*p840+p710*p01*p600+p830*p591*p00+p00*p631*p920+p640*p01*p680+p910*p671*p00+p00*p601*p840+p590*p01*p720+p830*p711*p00+p00*p581*p900+p570*p01*p650+p890*p661*p00+p00*p541*p860+p530*p01*p740+p850*p731*p00+p00*p561*p940+p550*p01*p610+p930*p621*p00+p00*p521*p880+p510*p01*p690+p870*p701*p00)-0.00696269*(p01*p610*p941+p621*p00*p561+p931*p550*p01+p01*p650*p901+p661*p00*p581+p891*p570*p01+p01*p690*p881+p701*p00*p521+p871*p510*p01+p01*p740*p861+p731*p00*p541+p851*p530*p01+p01*p680*p921+p671*p00*p631+p911*p640*p01+p01*p720*p841+p711*p00*p601+p831*p590*p01+p01*p630*p921+p641*p00*p681+p911*p670*p01+p01*p600*p841+p591*p00*p721+p831*p710*p01+p01*p580*p901+p571*p00*p651+p891*p660*p01+p01*p540*p861+p531*p00*p741+p851*p730*p01+p01*p560*p941+p551*p00*p611+p931*p620*p01+p01*p520*p881+p511*p00*p691+p871*p700*p01)+0.0225892*(p01*p611*p941+p621*p01*p561+p931*p551*p01+p01*p651*p901+p661*p01*p581+p891*p571*p01+p01*p691*p881+p701*p01*p521+p871*p511*p01+p01*p741*p861+p731*p01*p541+p851*p531*p01+p01*p681*p921+p671*p01*p631+p911*p641*p01+p01*p721*p841+p711*p01*p601+p831*p591*p01+p01*p631*p921+p641*p01*p681+p911*p671*p01+p01*p601*p841+p591*p01*p721+p831*p711*p01+p01*p581*p901+p571*p01*p651+p891*p661*p01+p01*p541*p861+p531*p01*p741+p851*p731*p01+p01*p561*p941+p551*p01*p611+p931*p621*p01+p01*p521*p881+p511*p01*p691+p871*p701*p01)+0.000247362*(p00*p610*p940+p620*p00*p560+p930*p550*p00+p00*p650*p900+p660*p00*p580+p890*p570*p00+p00*p690*p880+p700*p00*p520+p870*p510*p00+p00*p740*p860+p730*p00*p540+p850*p530*p00+p00*p680*p920+p670*p00*p630+p910*p640*p00+p00*p720*p840+p710*p00*p600+p830*p590*p00+p00*p630*p920+p640*p00*p680+p910*p670*p00+p00*p600*p840+p590*p00*p720+p830*p710*p00+p00*p580*p900+p570*p00*p650+p890*p660*p00+p00*p540*p860+p530*p00*p740+p850*p730*p00+p00*p560*p940+p550*p00*p610+p930*p620*p00+p00*p520*p880+p510*p00*p690+p870*p700*p00)-0.000567801*(p01*p840*p860+p831*p00*p930+p851*p940*p00+p01*p940*p850+p931*p00*p830+p861*p840*p00+p01*p930*p830+p941*p00*p850+p841*p860*p00+p01*p840*p880+p831*p00*p890+p871*p900*p00+p01*p900*p870+p891*p00*p830+p881*p840*p00+p01*p890*p830+p901*p00*p870+p841*p880*p00+p01*p840*p900+p831*p00*p870+p891*p880*p00+p01*p880*p890+p871*p00*p830+p901*p840*p00+p01*p870*p830+p881*p00*p890+p841*p900*p00+p01*p840*p940+p831*p00*p850+p931*p860*p00+p01*p860*p930+p851*p00*p830+p941*p840*p00+p01*p850*p830+p861*p00*p930+p841*p940*p00+p01*p860*p880+p851*p00*p910+p871*p920*p00+p01*p920*p870+p911*p00*p850+p881*p860*p00+p01*p910*p850+p921*p00*p870+p861*p880*p00+p01*p860*p920+p851*p00*p870+p911*p880*p00+p01*p880*p910+p871*p00*p850+p921*p860*p00+p01*p870*p850+p881*p00*p910+p861*p920*p00+p01*p900*p920+p891*p00*p930+p911*p940*p00+p01*p940*p910+p931*p00*p890+p921*p900*p00+p01*p930*p890+p941*p00*p910+p901*p920*p00+p01*p900*p940+p891*p00*p910+p931*p920*p00+p01*p920*p930+p911*p00*p890+p941*p900*p00+p01*p910*p890+p921*p00*p930+p901*p940*p00)-0.00165564*(p01*p841*p860+p831*p01*p930+p851*p941*p00+p01*p861*p840+p851*p01*p940+p831*p931*p00+p01*p941*p850+p931*p01*p830+p861*p841*p00+p01*p841*p880+p831*p01*p890+p871*p901*p00+p01*p881*p840+p871*p01*p900+p831*p891*p00+p01*p901*p870+p891*p01*p830+p881*p841*p00+p01*p841*p900+p831*p01*p870+p891*p881*p00+p01*p881*p890+p871*p01*p830+p901*p841*p00+p01*p901*p840+p891*p01*p880+p831*p871*p00+p01*p841*p940+p831*p01*p850+p931*p861*p00+p01*p861*p930+p851*p01*p830+p941*p841*p00+p01*p941*p840+p931*p01*p860+p831*p851*p00+p01*p861*p880+p851*p01*p910+p871*p921*p00+p01*p921*p870+p911*p01*p850+p881*p861*p00+p01*p881*p860+p871*p01*p920+p851*p911*p00+p01*p861*p920+p851*p01*p870+p911*p881*p00+p01*p921*p860+p911*p01*p880+p851*p871*p00+p01*p881*p910+p871*p01*p850+p921*p861*p00+p01*p901*p920+p891*p01*p930+p911*p941*p00+p01*p921*p900+p911*p01*p940+p891*p931*p00+p01*p941*p910+p931*p01*p890+p921*p901*p00+p01*p901*p940+p891*p01*p910+p931*p921*p00+p01*p921*p930+p911*p01*p890+p941*p901*p00+p01*p941*p900+p931*p01*p920+p891*p911*p00)-0.00432422*(p01*p841*p861+p831*p01*p931+p851*p941*p01+p01*p841*p881+p831*p01*p891+p871*p901*p01+p01*p841*p901+p831*p01*p871+p891*p881*p01+p01*p841*p941+p831*p01*p851+p931*p861*p01+p01*p861*p881+p851*p01*p911+p871*p921*p01+p01*p861*p921+p851*p01*p871+p911*p881*p01+p01*p901*p921+p891*p01*p931+p911*p941*p01+p01*p901*p941+p891*p01*p911+p931*p921*p01)-0.00603896*(p01*p90*p190*p150+p101*p00*p120*p140+p201*p110*p00*p170+p161*p130*p180*p00+p01*p110*p200*p170+p121*p00*p100*p140+p191*p90*p00*p150+p181*p130*p160*p00+p01*p130*p160*p180+p141*p00*p100*p120+p151*p90*p00*p190+p171*p110*p200*p00+p01*p140*p100*p120+p131*p00*p160*p180+p91*p150*p00*p190+p111*p170*p200*p00+p01*p90*p130*p110+p101*p00*p160*p200+p141*p150*p00*p170+p121*p190*p180*p00+p01*p150*p140*p170+p161*p00*p100*p200+p131*p90*p00*p110+p181*p190*p120*p00+p01*p190*p120*p180+p201*p00*p100*p160+p111*p90*p00*p130+p171*p150*p140*p00+p01*p200*p100*p160+p191*p00*p120*p180+p91*p110*p00*p130+p151*p170*p140*p00)-0.0270823*(p01*p91*p191*p150+p101*p01*p121*p140+p201*p111*p01*p170+p161*p131*p181*p00+p01*p91*p151*p190+p101*p01*p141*p120+p161*p131*p01*p180+p201*p111*p171*p00+p01*p111*p171*p200+p121*p01*p141*p100+p181*p131*p01*p160+p191*p91*p151*p00+p01*p151*p191*p90+p161*p01*p181*p130+p201*p171*p01*p110+p101*p141*p121*p00+p01*p91*p131*p110+p101*p01*p161*p200+p141*p151*p01*p170+p121*p191*p181*p00+p01*p91*p111*p130+p101*p01*p201*p160+p121*p191*p01*p180+p141*p151*p171*p00+p01*p111*p131*p90+p121*p01*p181*p190+p141*p171*p01*p150+p101*p201*p161*p00+p01*p151*p171*p140+p161*p01*p201*p100+p181*p191*p01*p120+p131*p91*p111*p00)-0.0486605*(p01*p91*p191*p151+p101*p01*p121*p141+p201*p111*p01*p171+p161*p131*p181*p01+p01*p91*p131*p111+p101*p01*p161*p201+p141*p151*p01*p171+p121*p191*p181*p01);
     return energy;
  }


  if(b == 1){
     l=index(i,j,k,1);
     int p00=mcL[l];
     l=index(i,j,k,1);
     int p01=mcL[l]*mcL[l];
     l=index(i,j,k,0);
     int p10=mcL[l];
     l=index(i,j,k,0);
     int p11=mcL[l]*mcL[l];
     l=index(i,j,k+1,0);
     int p20=mcL[l];
     l=index(i,j,k+1,0);
     int p21=mcL[l]*mcL[l];
     l=index(i,j+1,k,0);
     int p30=mcL[l];
     l=index(i,j+1,k,0);
     int p31=mcL[l]*mcL[l];
     l=index(i+1,j,k,0);
     int p40=mcL[l];
     l=index(i+1,j,k,0);
     int p41=mcL[l]*mcL[l];
     l=index(i,j,k-1,2);
     int p50=mcL[l];
     l=index(i,j,k-1,2);
     int p51=mcL[l]*mcL[l];
     l=index(i,j-1,k,2);
     int p60=mcL[l];
     l=index(i,j-1,k,2);
     int p61=mcL[l]*mcL[l];
     l=index(i-1,j-1,k,2);
     int p70=mcL[l];
     l=index(i-1,j-1,k,2);
     int p71=mcL[l]*mcL[l];
     l=index(i-1,j,k-1,2);
     int p80=mcL[l];
     l=index(i-1,j,k-1,2);
     int p81=mcL[l]*mcL[l];
     l=index(i,j-1,k-1,2);
     int p90=mcL[l];
     l=index(i,j-1,k-1,2);
     int p91=mcL[l]*mcL[l];
     l=index(i-1,j,k,2);
     int p100=mcL[l];
     l=index(i-1,j,k,2);
     int p101=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,2);
     int p110=mcL[l];
     l=index(i+1,j-1,k-1,2);
     int p111=mcL[l]*mcL[l];
     l=index(i-1,j-1,k-1,2);
     int p120=mcL[l];
     l=index(i-1,j-1,k-1,2);
     int p121=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,2);
     int p130=mcL[l];
     l=index(i-1,j-1,k+1,2);
     int p131=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,2);
     int p140=mcL[l];
     l=index(i-1,j+1,k-1,2);
     int p141=mcL[l]*mcL[l];
     l=index(i-1,j,k,0);
     int p150=mcL[l];
     l=index(i-1,j,k,0);
     int p151=mcL[l]*mcL[l];
     l=index(i-1,j,k+2,0);
     int p160=mcL[l];
     l=index(i-1,j,k+2,0);
     int p161=mcL[l]*mcL[l];
     l=index(i-1,j+2,k,0);
     int p170=mcL[l];
     l=index(i-1,j+2,k,0);
     int p171=mcL[l]*mcL[l];
     l=index(i,j-1,k,0);
     int p180=mcL[l];
     l=index(i,j-1,k,0);
     int p181=mcL[l]*mcL[l];
     l=index(i,j-1,k+2,0);
     int p190=mcL[l];
     l=index(i,j-1,k+2,0);
     int p191=mcL[l]*mcL[l];
     l=index(i,j,k-1,0);
     int p200=mcL[l];
     l=index(i,j,k-1,0);
     int p201=mcL[l]*mcL[l];
     l=index(i,j,k+2,0);
     int p210=mcL[l];
     l=index(i,j,k+2,0);
     int p211=mcL[l]*mcL[l];
     l=index(i,j+2,k-1,0);
     int p220=mcL[l];
     l=index(i,j+2,k-1,0);
     int p221=mcL[l]*mcL[l];
     l=index(i,j+2,k,0);
     int p230=mcL[l];
     l=index(i,j+2,k,0);
     int p231=mcL[l]*mcL[l];
     l=index(i+2,j-1,k,0);
     int p240=mcL[l];
     l=index(i+2,j-1,k,0);
     int p241=mcL[l]*mcL[l];
     l=index(i+2,j,k-1,0);
     int p250=mcL[l];
     l=index(i+2,j,k-1,0);
     int p251=mcL[l]*mcL[l];
     l=index(i+2,j,k,0);
     int p260=mcL[l];
     l=index(i+2,j,k,0);
     int p261=mcL[l]*mcL[l];
     l=index(i-2,j+1,k+1,0);
     int p270=mcL[l];
     l=index(i-2,j+1,k+1,0);
     int p271=mcL[l]*mcL[l];
     l=index(i+1,j-2,k+1,0);
     int p280=mcL[l];
     l=index(i+1,j-2,k+1,0);
     int p281=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-2,0);
     int p290=mcL[l];
     l=index(i+1,j+1,k-2,0);
     int p291=mcL[l]*mcL[l];
     l=index(i+1,j+1,k+1,0);
     int p300=mcL[l];
     l=index(i+1,j+1,k+1,0);
     int p301=mcL[l]*mcL[l];
     l=index(i+2,j,k,1);
     int p310=mcL[l];
     l=index(i+2,j,k,1);
     int p311=mcL[l]*mcL[l];
     l=index(i-2,j,k,1);
     int p320=mcL[l];
     l=index(i-2,j,k,1);
     int p321=mcL[l]*mcL[l];
     l=index(i+2,j,k-2,1);
     int p330=mcL[l];
     l=index(i+2,j,k-2,1);
     int p331=mcL[l]*mcL[l];
     l=index(i-2,j,k+2,1);
     int p340=mcL[l];
     l=index(i-2,j,k+2,1);
     int p341=mcL[l]*mcL[l];
     l=index(i+2,j-2,k,1);
     int p350=mcL[l];
     l=index(i+2,j-2,k,1);
     int p351=mcL[l]*mcL[l];
     l=index(i-2,j+2,k,1);
     int p360=mcL[l];
     l=index(i-2,j+2,k,1);
     int p361=mcL[l]*mcL[l];
     l=index(i,j+2,k,1);
     int p370=mcL[l];
     l=index(i,j+2,k,1);
     int p371=mcL[l]*mcL[l];
     l=index(i,j-2,k,1);
     int p380=mcL[l];
     l=index(i,j-2,k,1);
     int p381=mcL[l]*mcL[l];
     l=index(i,j+2,k-2,1);
     int p390=mcL[l];
     l=index(i,j+2,k-2,1);
     int p391=mcL[l]*mcL[l];
     l=index(i,j-2,k+2,1);
     int p400=mcL[l];
     l=index(i,j-2,k+2,1);
     int p401=mcL[l]*mcL[l];
     l=index(i,j,k+2,1);
     int p410=mcL[l];
     l=index(i,j,k+2,1);
     int p411=mcL[l]*mcL[l];
     l=index(i,j,k-2,1);
     int p420=mcL[l];
     l=index(i,j,k-2,1);
     int p421=mcL[l]*mcL[l];
     l=index(i,j+1,k+1,0);
     int p430=mcL[l];
     l=index(i,j+1,k+1,0);
     int p431=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p440=mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p441=mcL[l]*mcL[l];
     l=index(i,j+1,k-1,0);
     int p450=mcL[l];
     l=index(i,j+1,k-1,0);
     int p451=mcL[l]*mcL[l];
     l=index(i-1,j+1,k,0);
     int p460=mcL[l];
     l=index(i-1,j+1,k,0);
     int p461=mcL[l]*mcL[l];
     l=index(i,j-1,k+1,0);
     int p470=mcL[l];
     l=index(i,j-1,k+1,0);
     int p471=mcL[l]*mcL[l];
     l=index(i-1,j,k+1,0);
     int p480=mcL[l];
     l=index(i-1,j,k+1,0);
     int p481=mcL[l]*mcL[l];
     l=index(i+1,j,k+1,0);
     int p490=mcL[l];
     l=index(i+1,j,k+1,0);
     int p491=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p500=mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p501=mcL[l]*mcL[l];
     l=index(i+1,j,k-1,0);
     int p510=mcL[l];
     l=index(i+1,j,k-1,0);
     int p511=mcL[l]*mcL[l];
     l=index(i+1,j-1,k,0);
     int p520=mcL[l];
     l=index(i+1,j-1,k,0);
     int p521=mcL[l]*mcL[l];
     l=index(i+1,j+1,k,0);
     int p530=mcL[l];
     l=index(i+1,j+1,k,0);
     int p531=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p540=mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p541=mcL[l]*mcL[l];
     l=index(i+1,j+2,k-1,0);
     int p550=mcL[l];
     l=index(i+1,j+2,k-1,0);
     int p551=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+2,0);
     int p560=mcL[l];
     l=index(i+1,j-1,k+2,0);
     int p561=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,0);
     int p570=mcL[l];
     l=index(i+1,j-1,k-1,0);
     int p571=mcL[l]*mcL[l];
     l=index(i+2,j+1,k-1,0);
     int p580=mcL[l];
     l=index(i+2,j+1,k-1,0);
     int p581=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+2,0);
     int p590=mcL[l];
     l=index(i-1,j+1,k+2,0);
     int p591=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,0);
     int p600=mcL[l];
     l=index(i-1,j+1,k-1,0);
     int p601=mcL[l]*mcL[l];
     l=index(i+2,j-1,k+1,0);
     int p610=mcL[l];
     l=index(i+2,j-1,k+1,0);
     int p611=mcL[l]*mcL[l];
     l=index(i-1,j+2,k+1,0);
     int p620=mcL[l];
     l=index(i-1,j+2,k+1,0);
     int p621=mcL[l]*mcL[l];
     l=index(i+2,j-1,k-1,0);
     int p630=mcL[l];
     l=index(i+2,j-1,k-1,0);
     int p631=mcL[l]*mcL[l];
     l=index(i-1,j+2,k-1,0);
     int p640=mcL[l];
     l=index(i-1,j+2,k-1,0);
     int p641=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,0);
     int p650=mcL[l];
     l=index(i-1,j-1,k+1,0);
     int p651=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+2,0);
     int p660=mcL[l];
     l=index(i-1,j-1,k+2,0);
     int p661=mcL[l]*mcL[l];

     energy = energy-6.9586*(p01)+0.658756*(p00)-0.0346896*(p01*p10+p01*p20+p01*p30+p01*p40)-0.318022*(p00*p11+p00*p21+p00*p31+p00*p41)+1.35905*(p01*p11+p01*p21+p01*p31+p01*p41)+0.168277*(p00*p10+p00*p20+p00*p30+p00*p40)-0.184692*(p01*p50+p51*p00+p01*p60+p61*p00+p71*p00+p01*p70+p81*p00+p01*p80+p91*p00+p01*p90+p01*p100+p101*p00)+0.550731*(p01*p51+p01*p61+p71*p01+p81*p01+p91*p01+p01*p101)+0.11026*(p01*p111+p121*p01+p01*p131+p01*p141)-0.0276106*(p00*p110+p120*p00+p00*p130+p00*p140)-0.00322933*(p01*p151+p01*p161+p01*p171+p01*p181+p01*p191+p01*p201+p01*p211+p01*p221+p01*p231+p01*p241+p01*p251+p01*p261)-0.0766549*(p01*p270+p01*p280+p01*p290+p01*p300)+0.00623218*(p01*p271+p01*p281+p01*p291+p01*p301)+0.0842672*(p00*p270+p00*p280+p00*p290+p00*p300)+0.0162567*(p01*p310+p321*p00+p01*p320+p311*p00+p01*p330+p341*p00+p01*p340+p331*p00+p01*p350+p361*p00+p01*p360+p351*p00+p01*p370+p381*p00+p01*p380+p371*p00+p01*p390+p401*p00+p01*p400+p391*p00+p01*p410+p421*p00+p01*p420+p411*p00)+0.0191359*(p00*p310+p320*p00+p00*p330+p340*p00+p00*p350+p360*p00+p00*p370+p380*p00+p00*p390+p400*p00+p00*p410+p420*p00)-0.0135232*(p430*p440*p01+p450*p460*p01+p470*p480*p01+p490*p500*p01+p510*p520*p01+p530*p540*p01)-0.00637036*(p430*p440*p00+p450*p460*p00+p470*p480*p00+p490*p500*p00+p510*p520*p00+p530*p540*p00)-0.00476873*(p00*p40*p471+p00*p40*p451+p00*p40*p431+p00*p30*p481+p00*p30*p511+p00*p30*p491+p00*p20*p461+p00*p20*p521+p00*p10*p441+p00*p10*p501+p00*p20*p531+p00*p10*p541)-9.54342e-05*(p231*p170*p01+p171*p230*p01+p211*p160*p01+p161*p210*p01+p221*p170*p01+p171*p220*p01+p201*p150*p01+p151*p200*p01+p191*p160*p01+p161*p190*p01+p181*p150*p01+p151*p180*p01+p261*p240*p01+p241*p260*p01+p211*p190*p01+p191*p210*p01+p251*p240*p01+p241*p250*p01+p201*p180*p01+p181*p200*p01+p261*p250*p01+p251*p260*p01+p231*p220*p01+p221*p230*p01)+0.0234229*(p00*p531*p210+p00*p491*p230+p00*p541*p200+p00*p511*p220+p00*p501*p180+p00*p521*p190+p00*p431*p260+p00*p451*p250+p00*p471*p240+p00*p441*p150+p00*p461*p160+p00*p481*p170)-0.00137209*(p01*p40*p180+p01*p40*p200+p01*p40*p190+p01*p40*p210+p01*p40*p220+p01*p40*p230+p01*p30*p150+p01*p30*p200+p01*p30*p160+p01*p30*p210+p01*p20*p150+p01*p20*p180+p01*p10*p160+p01*p10*p190+p01*p20*p170+p01*p20*p230+p01*p10*p170+p01*p10*p220+p01*p30*p250+p01*p30*p260+p01*p20*p240+p01*p20*p260+p01*p10*p240+p01*p10*p250)+0.013012*(p141*p550*p00+p131*p560*p00+p01*p530*p130+p01*p490*p140+p121*p570*p00+p01*p540*p120+p01*p510*p140+p01*p500*p120+p01*p520*p130+p111*p580*p00+p01*p430*p110+p131*p590*p00+p01*p450*p110+p121*p600*p00+p111*p610*p00+p141*p620*p00+p111*p630*p00+p141*p640*p00+p01*p470*p110+p121*p650*p00+p131*p660*p00+p01*p440*p120+p01*p460*p130+p01*p480*p140)+0.0155687*(p140*p551*p00+p130*p561*p00+p00*p531*p130+p00*p491*p140+p120*p571*p00+p00*p541*p120+p00*p511*p140+p00*p501*p120+p00*p521*p130+p110*p581*p00+p00*p431*p110+p130*p591*p00+p00*p451*p110+p120*p601*p00+p110*p611*p00+p140*p621*p00+p110*p631*p00+p140*p641*p00+p00*p471*p110+p120*p651*p00+p130*p661*p00+p00*p441*p120+p00*p461*p130+p00*p481*p140)+0.0356897*(p140*p550*p01+p130*p560*p01+p00*p530*p131+p00*p490*p141+p120*p570*p01+p00*p540*p121+p00*p510*p141+p00*p500*p121+p00*p520*p131+p110*p580*p01+p00*p430*p111+p130*p590*p01+p00*p450*p111+p120*p600*p01+p110*p610*p01+p140*p620*p01+p110*p630*p01+p140*p640*p01+p00*p470*p111+p120*p650*p01+p130*p660*p01+p00*p440*p121+p00*p460*p131+p00*p480*p141)+0.0236792*(p221*p550*p00+p191*p560*p00+p231*p550*p00+p181*p570*p00+p211*p560*p00+p201*p570*p00+p251*p580*p00+p161*p590*p00+p261*p580*p00+p151*p600*p00+p241*p610*p00+p171*p620*p00+p241*p630*p00+p171*p640*p00+p261*p610*p00+p151*p650*p00+p251*p630*p00+p161*p660*p00+p211*p590*p00+p201*p600*p00+p231*p620*p00+p181*p650*p00+p221*p640*p00+p191*p660*p00)+0.0034611*(p220*p551*p00+p190*p561*p00+p230*p551*p00+p180*p571*p00+p210*p561*p00+p200*p571*p00+p250*p581*p00+p160*p591*p00+p260*p581*p00+p150*p601*p00+p240*p611*p00+p170*p621*p00+p240*p631*p00+p170*p641*p00+p260*p611*p00+p150*p651*p00+p250*p631*p00+p160*p661*p00+p210*p591*p00+p200*p601*p00+p230*p621*p00+p180*p651*p00+p220*p641*p00+p190*p661*p00)+0.0120949*(p00*p211*p260+p00*p261*p210+p00*p231*p260+p00*p261*p230+p00*p201*p250+p00*p251*p200+p00*p221*p250+p00*p251*p220+p00*p181*p240+p00*p241*p180+p00*p191*p240+p00*p241*p190+p00*p211*p230+p00*p231*p210+p00*p201*p220+p00*p221*p200+p00*p181*p190+p00*p191*p180+p00*p151*p170+p00*p171*p150+p00*p161*p170+p00*p171*p160+p00*p151*p160+p00*p161*p150);
     return energy;
  }


  if(b == 2){
     l=index(i,j,k,2);
     int p00=mcL[l];
     l=index(i,j,k,2);
     int p01=mcL[l]*mcL[l];
     l=index(i,j+1,k+1,0);
     int p10=mcL[l];
     l=index(i,j+1,k+1,0);
     int p11=mcL[l]*mcL[l];
     l=index(i+1,j,k+1,0);
     int p20=mcL[l];
     l=index(i+1,j,k+1,0);
     int p21=mcL[l]*mcL[l];
     l=index(i+1,j+1,k,0);
     int p30=mcL[l];
     l=index(i+1,j+1,k,0);
     int p31=mcL[l]*mcL[l];
     l=index(i+1,j+1,k+1,0);
     int p40=mcL[l];
     l=index(i+1,j+1,k+1,0);
     int p41=mcL[l]*mcL[l];
     l=index(i,j,k+1,1);
     int p50=mcL[l];
     l=index(i,j,k+1,1);
     int p51=mcL[l]*mcL[l];
     l=index(i,j+1,k,1);
     int p60=mcL[l];
     l=index(i,j+1,k,1);
     int p61=mcL[l]*mcL[l];
     l=index(i+1,j+1,k,1);
     int p70=mcL[l];
     l=index(i+1,j+1,k,1);
     int p71=mcL[l]*mcL[l];
     l=index(i+1,j,k+1,1);
     int p80=mcL[l];
     l=index(i+1,j,k+1,1);
     int p81=mcL[l]*mcL[l];
     l=index(i,j+1,k+1,1);
     int p90=mcL[l];
     l=index(i,j+1,k+1,1);
     int p91=mcL[l]*mcL[l];
     l=index(i+1,j,k,1);
     int p100=mcL[l];
     l=index(i+1,j,k,1);
     int p101=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,1);
     int p110=mcL[l];
     l=index(i-1,j+1,k+1,1);
     int p111=mcL[l]*mcL[l];
     l=index(i+1,j+1,k+1,1);
     int p120=mcL[l];
     l=index(i+1,j+1,k+1,1);
     int p121=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,1);
     int p130=mcL[l];
     l=index(i+1,j+1,k-1,1);
     int p131=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,1);
     int p140=mcL[l];
     l=index(i+1,j-1,k+1,1);
     int p141=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p150=mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p151=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+2,0);
     int p160=mcL[l];
     l=index(i-1,j+1,k+2,0);
     int p161=mcL[l]*mcL[l];
     l=index(i-1,j+2,k+1,0);
     int p170=mcL[l];
     l=index(i-1,j+2,k+1,0);
     int p171=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p180=mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p181=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+2,0);
     int p190=mcL[l];
     l=index(i+1,j-1,k+2,0);
     int p191=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p200=mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p201=mcL[l]*mcL[l];
     l=index(i+1,j+1,k+2,0);
     int p210=mcL[l];
     l=index(i+1,j+1,k+2,0);
     int p211=mcL[l]*mcL[l];
     l=index(i+1,j+2,k-1,0);
     int p220=mcL[l];
     l=index(i+1,j+2,k-1,0);
     int p221=mcL[l]*mcL[l];
     l=index(i+1,j+2,k+1,0);
     int p230=mcL[l];
     l=index(i+1,j+2,k+1,0);
     int p231=mcL[l]*mcL[l];
     l=index(i+2,j-1,k+1,0);
     int p240=mcL[l];
     l=index(i+2,j-1,k+1,0);
     int p241=mcL[l]*mcL[l];
     l=index(i+2,j+1,k-1,0);
     int p250=mcL[l];
     l=index(i+2,j+1,k-1,0);
     int p251=mcL[l]*mcL[l];
     l=index(i+2,j+1,k+1,0);
     int p260=mcL[l];
     l=index(i+2,j+1,k+1,0);
     int p261=mcL[l]*mcL[l];
     l=index(i,j,k,0);
     int p270=mcL[l];
     l=index(i,j,k,0);
     int p271=mcL[l]*mcL[l];
     l=index(i,j,k+3,0);
     int p280=mcL[l];
     l=index(i,j,k+3,0);
     int p281=mcL[l]*mcL[l];
     l=index(i,j+3,k,0);
     int p290=mcL[l];
     l=index(i,j+3,k,0);
     int p291=mcL[l]*mcL[l];
     l=index(i+3,j,k,0);
     int p300=mcL[l];
     l=index(i+3,j,k,0);
     int p301=mcL[l]*mcL[l];
     l=index(i+2,j,k,2);
     int p310=mcL[l];
     l=index(i+2,j,k,2);
     int p311=mcL[l]*mcL[l];
     l=index(i-2,j,k,2);
     int p320=mcL[l];
     l=index(i-2,j,k,2);
     int p321=mcL[l]*mcL[l];
     l=index(i+2,j,k-2,2);
     int p330=mcL[l];
     l=index(i+2,j,k-2,2);
     int p331=mcL[l]*mcL[l];
     l=index(i-2,j,k+2,2);
     int p340=mcL[l];
     l=index(i-2,j,k+2,2);
     int p341=mcL[l]*mcL[l];
     l=index(i+2,j-2,k,2);
     int p350=mcL[l];
     l=index(i+2,j-2,k,2);
     int p351=mcL[l]*mcL[l];
     l=index(i-2,j+2,k,2);
     int p360=mcL[l];
     l=index(i-2,j+2,k,2);
     int p361=mcL[l]*mcL[l];
     l=index(i,j+2,k,2);
     int p370=mcL[l];
     l=index(i,j+2,k,2);
     int p371=mcL[l]*mcL[l];
     l=index(i,j-2,k,2);
     int p380=mcL[l];
     l=index(i,j-2,k,2);
     int p381=mcL[l]*mcL[l];
     l=index(i,j+2,k-2,2);
     int p390=mcL[l];
     l=index(i,j+2,k-2,2);
     int p391=mcL[l]*mcL[l];
     l=index(i,j-2,k+2,2);
     int p400=mcL[l];
     l=index(i,j-2,k+2,2);
     int p401=mcL[l]*mcL[l];
     l=index(i,j,k+2,2);
     int p410=mcL[l];
     l=index(i,j,k+2,2);
     int p411=mcL[l]*mcL[l];
     l=index(i,j,k-2,2);
     int p420=mcL[l];
     l=index(i,j,k-2,2);
     int p421=mcL[l]*mcL[l];
     l=index(i+2,j,k,0);
     int p430=mcL[l];
     l=index(i+2,j,k,0);
     int p431=mcL[l]*mcL[l];
     l=index(i+1,j,k,0);
     int p440=mcL[l];
     l=index(i+1,j,k,0);
     int p441=mcL[l]*mcL[l];
     l=index(i+2,j,k+1,0);
     int p450=mcL[l];
     l=index(i+2,j,k+1,0);
     int p451=mcL[l]*mcL[l];
     l=index(i+1,j,k+2,0);
     int p460=mcL[l];
     l=index(i+1,j,k+2,0);
     int p461=mcL[l]*mcL[l];
     l=index(i+2,j+1,k,0);
     int p470=mcL[l];
     l=index(i+2,j+1,k,0);
     int p471=mcL[l]*mcL[l];
     l=index(i+1,j+2,k,0);
     int p480=mcL[l];
     l=index(i+1,j+2,k,0);
     int p481=mcL[l]*mcL[l];
     l=index(i,j+2,k,0);
     int p490=mcL[l];
     l=index(i,j+2,k,0);
     int p491=mcL[l]*mcL[l];
     l=index(i,j+1,k,0);
     int p500=mcL[l];
     l=index(i,j+1,k,0);
     int p501=mcL[l]*mcL[l];
     l=index(i,j+2,k+1,0);
     int p510=mcL[l];
     l=index(i,j+2,k+1,0);
     int p511=mcL[l]*mcL[l];
     l=index(i,j+1,k+2,0);
     int p520=mcL[l];
     l=index(i,j+1,k+2,0);
     int p521=mcL[l]*mcL[l];
     l=index(i,j,k+2,0);
     int p530=mcL[l];
     l=index(i,j,k+2,0);
     int p531=mcL[l]*mcL[l];
     l=index(i,j,k+1,0);
     int p540=mcL[l];
     l=index(i,j,k+1,0);
     int p541=mcL[l]*mcL[l];
     l=index(i+2,j+2,k-1,0);
     int p550=mcL[l];
     l=index(i+2,j+2,k-1,0);
     int p551=mcL[l]*mcL[l];
     l=index(i+2,j-1,k+2,0);
     int p560=mcL[l];
     l=index(i+2,j-1,k+2,0);
     int p561=mcL[l]*mcL[l];
     l=index(i+2,j+2,k,0);
     int p570=mcL[l];
     l=index(i+2,j+2,k,0);
     int p571=mcL[l]*mcL[l];
     l=index(i+2,j-1,k,0);
     int p580=mcL[l];
     l=index(i+2,j-1,k,0);
     int p581=mcL[l]*mcL[l];
     l=index(i+2,j,k+2,0);
     int p590=mcL[l];
     l=index(i+2,j,k+2,0);
     int p591=mcL[l]*mcL[l];
     l=index(i+2,j,k-1,0);
     int p600=mcL[l];
     l=index(i+2,j,k-1,0);
     int p601=mcL[l]*mcL[l];
     l=index(i-1,j+2,k+2,0);
     int p610=mcL[l];
     l=index(i-1,j+2,k+2,0);
     int p611=mcL[l]*mcL[l];
     l=index(i-1,j+2,k,0);
     int p620=mcL[l];
     l=index(i-1,j+2,k,0);
     int p621=mcL[l]*mcL[l];
     l=index(i-1,j,k+2,0);
     int p630=mcL[l];
     l=index(i-1,j,k+2,0);
     int p631=mcL[l]*mcL[l];
     l=index(i,j+2,k+2,0);
     int p640=mcL[l];
     l=index(i,j+2,k+2,0);
     int p641=mcL[l]*mcL[l];
     l=index(i,j+2,k-1,0);
     int p650=mcL[l];
     l=index(i,j+2,k-1,0);
     int p651=mcL[l]*mcL[l];
     l=index(i,j-1,k+2,0);
     int p660=mcL[l];
     l=index(i,j-1,k+2,0);
     int p661=mcL[l]*mcL[l];

     energy = energy-6.9586*(p01)+0.658756*(p00)-0.0346896*(p01*p10+p01*p20+p01*p30+p01*p40)-0.318022*(p00*p11+p00*p21+p00*p31+p00*p41)+1.35905*(p01*p11+p01*p21+p01*p31+p01*p41)+0.168277*(p00*p10+p00*p20+p00*p30+p00*p40)-0.184692*(p51*p00+p01*p50+p61*p00+p01*p60+p01*p70+p71*p00+p01*p80+p81*p00+p01*p90+p91*p00+p101*p00+p01*p100)+0.550731*(p51*p01+p61*p01+p01*p71+p01*p81+p01*p91+p101*p01)+0.11026*(p111*p01+p01*p121+p131*p01+p141*p01)-0.0276106*(p110*p00+p00*p120+p130*p00+p140*p00)-0.00322933*(p01*p151+p01*p161+p01*p171+p01*p181+p01*p191+p01*p201+p01*p211+p01*p221+p01*p231+p01*p241+p01*p251+p01*p261)-0.0766549*(p01*p270+p01*p280+p01*p290+p01*p300)+0.00623218*(p01*p271+p01*p281+p01*p291+p01*p301)+0.0842672*(p00*p270+p00*p280+p00*p290+p00*p300)+0.0162567*(p01*p310+p321*p00+p01*p320+p311*p00+p01*p330+p341*p00+p01*p340+p331*p00+p01*p350+p361*p00+p01*p360+p351*p00+p01*p370+p381*p00+p01*p380+p371*p00+p01*p390+p401*p00+p01*p400+p391*p00+p01*p410+p421*p00+p01*p420+p411*p00)+0.0191359*(p00*p310+p320*p00+p00*p330+p340*p00+p00*p350+p360*p00+p00*p370+p380*p00+p00*p390+p400*p00+p00*p410+p420*p00)-0.0135232*(p430*p440*p01+p450*p460*p01+p470*p480*p01+p490*p500*p01+p510*p520*p01+p530*p540*p01)-0.00637036*(p430*p440*p00+p450*p460*p00+p470*p480*p00+p490*p500*p00+p510*p520*p00+p530*p540*p00)-0.00476873*(p00*p40*p531+p00*p40*p491+p00*p30*p541+p00*p30*p511+p00*p20*p501+p00*p20*p521+p00*p40*p431+p00*p30*p451+p00*p20*p471+p00*p10*p441+p00*p10*p461+p00*p10*p481)-9.54342e-05*(p251*p200*p01+p201*p250*p01+p241*p180*p01+p181*p240*p01+p261*p210*p01+p211*p260*p01+p241*p190*p01+p191*p240*p01+p261*p230*p01+p231*p260*p01+p251*p220*p01+p221*p250*p01+p221*p200*p01+p201*p220*p01+p171*p150*p01+p151*p170*p01+p231*p210*p01+p211*p230*p01+p171*p160*p01+p161*p170*p01+p191*p180*p01+p181*p190*p01+p161*p150*p01+p151*p160*p01)+0.0234229*(p00*p471*p240+p00*p451*p250+p00*p431*p260+p00*p481*p170+p00*p511*p220+p00*p491*p230+p00*p461*p160+p00*p521*p190+p00*p441*p150+p00*p501*p180+p00*p531*p210+p00*p541*p200)-0.00137209*(p01*p40*p160+p01*p40*p170+p01*p30*p150+p01*p30*p170+p01*p20*p150+p01*p20*p160+p01*p40*p190+p01*p40*p240+p01*p30*p180+p01*p30*p240+p01*p40*p220+p01*p40*p250+p01*p30*p230+p01*p30*p260+p01*p20*p200+p01*p20*p250+p01*p20*p210+p01*p20*p260+p01*p10*p180+p01*p10*p190+p01*p10*p200+p01*p10*p220+p01*p10*p210+p01*p10*p230)+0.013012*(p01*p470*p140+p01*p450*p130+p131*p550*p00+p141*p560*p00+p01*p430*p120+p121*p570*p00+p141*p580*p00+p121*p590*p00+p131*p600*p00+p01*p480*p110+p111*p610*p00+p01*p510*p130+p111*p620*p00+p01*p490*p120+p01*p460*p110+p01*p520*p140+p01*p440*p110+p01*p500*p140+p111*p630*p00+p01*p530*p120+p01*p540*p130+p121*p640*p00+p131*p650*p00+p141*p660*p00)+0.0155687*(p00*p471*p140+p00*p451*p130+p130*p551*p00+p140*p561*p00+p00*p431*p120+p120*p571*p00+p140*p581*p00+p120*p591*p00+p130*p601*p00+p00*p481*p110+p110*p611*p00+p00*p511*p130+p110*p621*p00+p00*p491*p120+p00*p461*p110+p00*p521*p140+p00*p441*p110+p00*p501*p140+p110*p631*p00+p00*p531*p120+p00*p541*p130+p120*p641*p00+p130*p651*p00+p140*p661*p00)+0.0356897*(p00*p470*p141+p00*p450*p131+p130*p550*p01+p140*p560*p01+p00*p430*p121+p120*p570*p01+p140*p580*p01+p120*p590*p01+p130*p600*p01+p00*p480*p111+p110*p610*p01+p00*p510*p131+p110*p620*p01+p00*p490*p121+p00*p460*p111+p00*p520*p141+p00*p440*p111+p00*p500*p141+p110*p630*p01+p00*p530*p121+p00*p540*p131+p120*p640*p01+p130*p650*p01+p140*p660*p01)+0.0236792*(p221*p550*p00+p191*p560*p00+p231*p570*p00+p181*p580*p00+p211*p590*p00+p201*p600*p00+p251*p550*p00+p161*p610*p00+p261*p570*p00+p151*p620*p00+p241*p560*p00+p171*p610*p00+p241*p580*p00+p171*p620*p00+p261*p590*p00+p151*p630*p00+p251*p600*p00+p161*p630*p00+p211*p640*p00+p201*p650*p00+p231*p640*p00+p181*p660*p00+p221*p650*p00+p191*p660*p00)+0.0034611*(p220*p551*p00+p190*p561*p00+p230*p571*p00+p180*p581*p00+p210*p591*p00+p200*p601*p00+p250*p551*p00+p160*p611*p00+p260*p571*p00+p150*p621*p00+p240*p561*p00+p170*p611*p00+p240*p581*p00+p170*p621*p00+p260*p591*p00+p150*p631*p00+p250*p601*p00+p160*p631*p00+p210*p641*p00+p200*p651*p00+p230*p641*p00+p180*p661*p00+p220*p651*p00+p190*p661*p00)+0.0120949*(p00*p241*p260+p00*p261*p240+p00*p251*p260+p00*p261*p250+p00*p241*p250+p00*p251*p240+p00*p171*p230+p00*p231*p170+p00*p221*p230+p00*p231*p220+p00*p171*p220+p00*p221*p170+p00*p161*p210+p00*p211*p160+p00*p191*p210+p00*p211*p190+p00*p151*p200+p00*p201*p150+p00*p181*p200+p00*p201*p180+p00*p161*p190+p00*p191*p160+p00*p151*p180+p00*p181*p150);
     return energy;
  }


}


double Monte_Carlo::normalized_pointenergy(int i, int j, int k, int b){
  double energy = 0.0;
  int l; 
  if(b == 0){
     l=index(i,j,k,0);
     int p00=mcL[l];
     l=index(i,j,k,0);
     int p01=mcL[l]*mcL[l];
     l=index(i,j-1,k-1,2);
     int p10=mcL[l];
     l=index(i,j-1,k-1,2);
     int p11=mcL[l]*mcL[l];
     l=index(i,j,k,1);
     int p20=mcL[l];
     l=index(i,j,k,1);
     int p21=mcL[l]*mcL[l];
     l=index(i,j,k-1,1);
     int p30=mcL[l];
     l=index(i,j,k-1,1);
     int p31=mcL[l]*mcL[l];
     l=index(i,j-1,k,1);
     int p40=mcL[l];
     l=index(i,j-1,k,1);
     int p41=mcL[l]*mcL[l];
     l=index(i-1,j,k-1,2);
     int p50=mcL[l];
     l=index(i-1,j,k-1,2);
     int p51=mcL[l]*mcL[l];
     l=index(i-1,j-1,k,2);
     int p60=mcL[l];
     l=index(i-1,j-1,k,2);
     int p61=mcL[l]*mcL[l];
     l=index(i-1,j-1,k-1,2);
     int p70=mcL[l];
     l=index(i-1,j-1,k-1,2);
     int p71=mcL[l]*mcL[l];
     l=index(i-1,j,k,1);
     int p80=mcL[l];
     l=index(i-1,j,k,1);
     int p81=mcL[l]*mcL[l];
     l=index(i+1,j,k,0);
     int p90=mcL[l];
     l=index(i+1,j,k,0);
     int p91=mcL[l]*mcL[l];
     l=index(i-1,j,k,0);
     int p100=mcL[l];
     l=index(i-1,j,k,0);
     int p101=mcL[l]*mcL[l];
     l=index(i+1,j,k-1,0);
     int p110=mcL[l];
     l=index(i+1,j,k-1,0);
     int p111=mcL[l]*mcL[l];
     l=index(i-1,j,k+1,0);
     int p120=mcL[l];
     l=index(i-1,j,k+1,0);
     int p121=mcL[l]*mcL[l];
     l=index(i+1,j-1,k,0);
     int p130=mcL[l];
     l=index(i+1,j-1,k,0);
     int p131=mcL[l]*mcL[l];
     l=index(i-1,j+1,k,0);
     int p140=mcL[l];
     l=index(i-1,j+1,k,0);
     int p141=mcL[l]*mcL[l];
     l=index(i,j+1,k,0);
     int p150=mcL[l];
     l=index(i,j+1,k,0);
     int p151=mcL[l]*mcL[l];
     l=index(i,j-1,k,0);
     int p160=mcL[l];
     l=index(i,j-1,k,0);
     int p161=mcL[l]*mcL[l];
     l=index(i,j+1,k-1,0);
     int p170=mcL[l];
     l=index(i,j+1,k-1,0);
     int p171=mcL[l]*mcL[l];
     l=index(i,j-1,k+1,0);
     int p180=mcL[l];
     l=index(i,j-1,k+1,0);
     int p181=mcL[l]*mcL[l];
     l=index(i,j,k+1,0);
     int p190=mcL[l];
     l=index(i,j,k+1,0);
     int p191=mcL[l]*mcL[l];
     l=index(i,j,k-1,0);
     int p200=mcL[l];
     l=index(i,j,k-1,0);
     int p201=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p210=mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p211=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,0);
     int p220=mcL[l];
     l=index(i-1,j-1,k+1,0);
     int p221=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p230=mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p231=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,0);
     int p240=mcL[l];
     l=index(i-1,j+1,k-1,0);
     int p241=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p250=mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p251=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,0);
     int p260=mcL[l];
     l=index(i+1,j-1,k-1,0);
     int p261=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,2);
     int p270=mcL[l];
     l=index(i+1,j-1,k-1,2);
     int p271=mcL[l]*mcL[l];
     l=index(i+1,j,k,1);
     int p280=mcL[l];
     l=index(i+1,j,k,1);
     int p281=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-2,2);
     int p290=mcL[l];
     l=index(i+1,j-1,k-2,2);
     int p291=mcL[l]*mcL[l];
     l=index(i+1,j,k-2,1);
     int p300=mcL[l];
     l=index(i+1,j,k-2,1);
     int p301=mcL[l]*mcL[l];
     l=index(i+1,j-2,k-1,2);
     int p310=mcL[l];
     l=index(i+1,j-2,k-1,2);
     int p311=mcL[l]*mcL[l];
     l=index(i+1,j-2,k,1);
     int p320=mcL[l];
     l=index(i+1,j-2,k,1);
     int p321=mcL[l]*mcL[l];
     l=index(i,j+1,k,1);
     int p330=mcL[l];
     l=index(i,j+1,k,1);
     int p331=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,2);
     int p340=mcL[l];
     l=index(i-1,j+1,k-1,2);
     int p341=mcL[l]*mcL[l];
     l=index(i,j+1,k-2,1);
     int p350=mcL[l];
     l=index(i,j+1,k-2,1);
     int p351=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-2,2);
     int p360=mcL[l];
     l=index(i-1,j+1,k-2,2);
     int p361=mcL[l]*mcL[l];
     l=index(i,j,k+1,1);
     int p370=mcL[l];
     l=index(i,j,k+1,1);
     int p371=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,2);
     int p380=mcL[l];
     l=index(i-1,j-1,k+1,2);
     int p381=mcL[l]*mcL[l];
     l=index(i,j,k-2,1);
     int p390=mcL[l];
     l=index(i,j,k-2,1);
     int p391=mcL[l]*mcL[l];
     l=index(i-1,j-1,k-2,2);
     int p400=mcL[l];
     l=index(i-1,j-1,k-2,2);
     int p401=mcL[l]*mcL[l];
     l=index(i,j-2,k+1,1);
     int p410=mcL[l];
     l=index(i,j-2,k+1,1);
     int p411=mcL[l]*mcL[l];
     l=index(i-1,j-2,k+1,2);
     int p420=mcL[l];
     l=index(i-1,j-2,k+1,2);
     int p421=mcL[l]*mcL[l];
     l=index(i,j-2,k,1);
     int p430=mcL[l];
     l=index(i,j-2,k,1);
     int p431=mcL[l]*mcL[l];
     l=index(i-1,j-2,k-1,2);
     int p440=mcL[l];
     l=index(i-1,j-2,k-1,2);
     int p441=mcL[l]*mcL[l];
     l=index(i-2,j+1,k-1,2);
     int p450=mcL[l];
     l=index(i-2,j+1,k-1,2);
     int p451=mcL[l]*mcL[l];
     l=index(i-2,j+1,k,1);
     int p460=mcL[l];
     l=index(i-2,j+1,k,1);
     int p461=mcL[l]*mcL[l];
     l=index(i-2,j-1,k+1,2);
     int p470=mcL[l];
     l=index(i-2,j-1,k+1,2);
     int p471=mcL[l]*mcL[l];
     l=index(i-2,j,k+1,1);
     int p480=mcL[l];
     l=index(i-2,j,k+1,1);
     int p481=mcL[l]*mcL[l];
     l=index(i-2,j-1,k-1,2);
     int p490=mcL[l];
     l=index(i-2,j-1,k-1,2);
     int p491=mcL[l]*mcL[l];
     l=index(i-2,j,k,1);
     int p500=mcL[l];
     l=index(i-2,j,k,1);
     int p501=mcL[l]*mcL[l];
     l=index(i+2,j,k-1,0);
     int p510=mcL[l];
     l=index(i+2,j,k-1,0);
     int p511=mcL[l]*mcL[l];
     l=index(i-2,j,k+1,0);
     int p520=mcL[l];
     l=index(i-2,j,k+1,0);
     int p521=mcL[l]*mcL[l];
     l=index(i+2,j-1,k,0);
     int p530=mcL[l];
     l=index(i+2,j-1,k,0);
     int p531=mcL[l]*mcL[l];
     l=index(i-2,j+1,k,0);
     int p540=mcL[l];
     l=index(i-2,j+1,k,0);
     int p541=mcL[l]*mcL[l];
     l=index(i+1,j+1,k,0);
     int p550=mcL[l];
     l=index(i+1,j+1,k,0);
     int p551=mcL[l]*mcL[l];
     l=index(i-1,j-1,k,0);
     int p560=mcL[l];
     l=index(i-1,j-1,k,0);
     int p561=mcL[l]*mcL[l];
     l=index(i+1,j,k+1,0);
     int p570=mcL[l];
     l=index(i+1,j,k+1,0);
     int p571=mcL[l]*mcL[l];
     l=index(i-1,j,k-1,0);
     int p580=mcL[l];
     l=index(i-1,j,k-1,0);
     int p581=mcL[l]*mcL[l];
     l=index(i+2,j-1,k-1,0);
     int p590=mcL[l];
     l=index(i+2,j-1,k-1,0);
     int p591=mcL[l]*mcL[l];
     l=index(i-2,j+1,k+1,0);
     int p600=mcL[l];
     l=index(i-2,j+1,k+1,0);
     int p601=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-2,0);
     int p610=mcL[l];
     l=index(i+1,j+1,k-2,0);
     int p611=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+2,0);
     int p620=mcL[l];
     l=index(i-1,j-1,k+2,0);
     int p621=mcL[l]*mcL[l];
     l=index(i-1,j,k+2,0);
     int p630=mcL[l];
     l=index(i-1,j,k+2,0);
     int p631=mcL[l]*mcL[l];
     l=index(i+1,j,k-2,0);
     int p640=mcL[l];
     l=index(i+1,j,k-2,0);
     int p641=mcL[l]*mcL[l];
     l=index(i+1,j-2,k+1,0);
     int p650=mcL[l];
     l=index(i+1,j-2,k+1,0);
     int p651=mcL[l]*mcL[l];
     l=index(i-1,j+2,k-1,0);
     int p660=mcL[l];
     l=index(i-1,j+2,k-1,0);
     int p661=mcL[l]*mcL[l];
     l=index(i-1,j+2,k,0);
     int p670=mcL[l];
     l=index(i-1,j+2,k,0);
     int p671=mcL[l]*mcL[l];
     l=index(i+1,j-2,k,0);
     int p680=mcL[l];
     l=index(i+1,j-2,k,0);
     int p681=mcL[l]*mcL[l];
     l=index(i,j+2,k-1,0);
     int p690=mcL[l];
     l=index(i,j+2,k-1,0);
     int p691=mcL[l]*mcL[l];
     l=index(i,j-2,k+1,0);
     int p700=mcL[l];
     l=index(i,j-2,k+1,0);
     int p701=mcL[l]*mcL[l];
     l=index(i,j+1,k+1,0);
     int p710=mcL[l];
     l=index(i,j+1,k+1,0);
     int p711=mcL[l]*mcL[l];
     l=index(i,j-1,k-1,0);
     int p720=mcL[l];
     l=index(i,j-1,k-1,0);
     int p721=mcL[l]*mcL[l];
     l=index(i,j+1,k-2,0);
     int p730=mcL[l];
     l=index(i,j+1,k-2,0);
     int p731=mcL[l]*mcL[l];
     l=index(i,j-1,k+2,0);
     int p740=mcL[l];
     l=index(i,j-1,k+2,0);
     int p741=mcL[l]*mcL[l];
     l=index(i,j,k,2);
     int p750=mcL[l];
     l=index(i,j,k,2);
     int p751=mcL[l]*mcL[l];
     l=index(i+2,j-1,k-1,1);
     int p760=mcL[l];
     l=index(i+2,j-1,k-1,1);
     int p761=mcL[l]*mcL[l];
     l=index(i,j,k-3,2);
     int p770=mcL[l];
     l=index(i,j,k-3,2);
     int p771=mcL[l]*mcL[l];
     l=index(i,j-3,k,2);
     int p780=mcL[l];
     l=index(i,j-3,k,2);
     int p781=mcL[l]*mcL[l];
     l=index(i-1,j+2,k-1,1);
     int p790=mcL[l];
     l=index(i-1,j+2,k-1,1);
     int p791=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+2,1);
     int p800=mcL[l];
     l=index(i-1,j-1,k+2,1);
     int p801=mcL[l]*mcL[l];
     l=index(i-1,j-1,k-1,1);
     int p810=mcL[l];
     l=index(i-1,j-1,k-1,1);
     int p811=mcL[l]*mcL[l];
     l=index(i-3,j,k,2);
     int p820=mcL[l];
     l=index(i-3,j,k,2);
     int p821=mcL[l]*mcL[l];
     l=index(i+2,j,k,0);
     int p830=mcL[l];
     l=index(i+2,j,k,0);
     int p831=mcL[l]*mcL[l];
     l=index(i-2,j,k,0);
     int p840=mcL[l];
     l=index(i-2,j,k,0);
     int p841=mcL[l]*mcL[l];
     l=index(i+2,j,k-2,0);
     int p850=mcL[l];
     l=index(i+2,j,k-2,0);
     int p851=mcL[l]*mcL[l];
     l=index(i-2,j,k+2,0);
     int p860=mcL[l];
     l=index(i-2,j,k+2,0);
     int p861=mcL[l]*mcL[l];
     l=index(i+2,j-2,k,0);
     int p870=mcL[l];
     l=index(i+2,j-2,k,0);
     int p871=mcL[l]*mcL[l];
     l=index(i-2,j+2,k,0);
     int p880=mcL[l];
     l=index(i-2,j+2,k,0);
     int p881=mcL[l]*mcL[l];
     l=index(i,j+2,k,0);
     int p890=mcL[l];
     l=index(i,j+2,k,0);
     int p891=mcL[l]*mcL[l];
     l=index(i,j-2,k,0);
     int p900=mcL[l];
     l=index(i,j-2,k,0);
     int p901=mcL[l]*mcL[l];
     l=index(i,j+2,k-2,0);
     int p910=mcL[l];
     l=index(i,j+2,k-2,0);
     int p911=mcL[l]*mcL[l];
     l=index(i,j-2,k+2,0);
     int p920=mcL[l];
     l=index(i,j-2,k+2,0);
     int p921=mcL[l]*mcL[l];
     l=index(i,j,k+2,0);
     int p930=mcL[l];
     l=index(i,j,k+2,0);
     int p931=mcL[l]*mcL[l];
     l=index(i,j,k-2,0);
     int p940=mcL[l];
     l=index(i,j,k-2,0);
     int p941=mcL[l]*mcL[l];
     l=index(i-2,j,k,2);
     int p950=mcL[l];
     l=index(i-2,j,k,2);
     int p951=mcL[l]*mcL[l];
     l=index(i-1,j,k,2);
     int p960=mcL[l];
     l=index(i-1,j,k,2);
     int p961=mcL[l]*mcL[l];
     l=index(i,j-1,k-1,1);
     int p970=mcL[l];
     l=index(i,j-1,k-1,1);
     int p971=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,1);
     int p980=mcL[l];
     l=index(i+1,j-1,k-1,1);
     int p981=mcL[l]*mcL[l];
     l=index(i-2,j,k-1,2);
     int p990=mcL[l];
     l=index(i-2,j,k-1,2);
     int p991=mcL[l]*mcL[l];
     l=index(i-1,j,k-2,2);
     int p1000=mcL[l];
     l=index(i-1,j,k-2,2);
     int p1001=mcL[l]*mcL[l];
     l=index(i,j-1,k+1,1);
     int p1010=mcL[l];
     l=index(i,j-1,k+1,1);
     int p1011=mcL[l]*mcL[l];
     l=index(i+1,j-1,k,1);
     int p1020=mcL[l];
     l=index(i+1,j-1,k,1);
     int p1021=mcL[l]*mcL[l];
     l=index(i-2,j-1,k,2);
     int p1030=mcL[l];
     l=index(i-2,j-1,k,2);
     int p1031=mcL[l]*mcL[l];
     l=index(i-1,j-2,k,2);
     int p1040=mcL[l];
     l=index(i-1,j-2,k,2);
     int p1041=mcL[l]*mcL[l];
     l=index(i,j+1,k-1,1);
     int p1050=mcL[l];
     l=index(i,j+1,k-1,1);
     int p1051=mcL[l]*mcL[l];
     l=index(i+1,j,k-1,1);
     int p1060=mcL[l];
     l=index(i+1,j,k-1,1);
     int p1061=mcL[l]*mcL[l];
     l=index(i-1,j,k-1,1);
     int p1070=mcL[l];
     l=index(i-1,j,k-1,1);
     int p1071=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,1);
     int p1080=mcL[l];
     l=index(i-1,j+1,k-1,1);
     int p1081=mcL[l]*mcL[l];
     l=index(i,j-2,k,2);
     int p1090=mcL[l];
     l=index(i,j-2,k,2);
     int p1091=mcL[l]*mcL[l];
     l=index(i,j-1,k,2);
     int p1100=mcL[l];
     l=index(i,j-1,k,2);
     int p1101=mcL[l]*mcL[l];
     l=index(i-1,j,k+1,1);
     int p1110=mcL[l];
     l=index(i-1,j,k+1,1);
     int p1111=mcL[l]*mcL[l];
     l=index(i-1,j+1,k,1);
     int p1120=mcL[l];
     l=index(i-1,j+1,k,1);
     int p1121=mcL[l]*mcL[l];
     l=index(i,j-2,k-1,2);
     int p1130=mcL[l];
     l=index(i,j-2,k-1,2);
     int p1131=mcL[l]*mcL[l];
     l=index(i,j-1,k-2,2);
     int p1140=mcL[l];
     l=index(i,j-1,k-2,2);
     int p1141=mcL[l]*mcL[l];
     l=index(i-1,j-1,k,1);
     int p1150=mcL[l];
     l=index(i-1,j-1,k,1);
     int p1151=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,1);
     int p1160=mcL[l];
     l=index(i-1,j-1,k+1,1);
     int p1161=mcL[l]*mcL[l];
     l=index(i,j,k-2,2);
     int p1170=mcL[l];
     l=index(i,j,k-2,2);
     int p1171=mcL[l]*mcL[l];
     l=index(i,j,k-1,2);
     int p1180=mcL[l];
     l=index(i,j,k-1,2);
     int p1181=mcL[l]*mcL[l];
     l=index(i-1,j-2,k+1,1);
     int p1190=mcL[l];
     l=index(i-1,j-2,k+1,1);
     int p1191=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-2,1);
     int p1200=mcL[l];
     l=index(i-1,j+1,k-2,1);
     int p1201=mcL[l]*mcL[l];
     l=index(i-2,j-2,k+1,2);
     int p1210=mcL[l];
     l=index(i-2,j-2,k+1,2);
     int p1211=mcL[l]*mcL[l];
     l=index(i-2,j+1,k-2,2);
     int p1220=mcL[l];
     l=index(i-2,j+1,k-2,2);
     int p1221=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,1);
     int p1230=mcL[l];
     l=index(i-1,j+1,k+1,1);
     int p1231=mcL[l]*mcL[l];
     l=index(i-2,j-2,k,2);
     int p1240=mcL[l];
     l=index(i-2,j-2,k,2);
     int p1241=mcL[l]*mcL[l];
     l=index(i-2,j+1,k,2);
     int p1250=mcL[l];
     l=index(i-2,j+1,k,2);
     int p1251=mcL[l]*mcL[l];
     l=index(i-2,j,k-2,2);
     int p1260=mcL[l];
     l=index(i-2,j,k-2,2);
     int p1261=mcL[l]*mcL[l];
     l=index(i-2,j,k+1,2);
     int p1270=mcL[l];
     l=index(i-2,j,k+1,2);
     int p1271=mcL[l]*mcL[l];
     l=index(i-2,j-1,k+1,1);
     int p1280=mcL[l];
     l=index(i-2,j-1,k+1,1);
     int p1281=mcL[l]*mcL[l];
     l=index(i+1,j-2,k-2,2);
     int p1290=mcL[l];
     l=index(i+1,j-2,k-2,2);
     int p1291=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-2,1);
     int p1300=mcL[l];
     l=index(i+1,j-1,k-2,1);
     int p1301=mcL[l]*mcL[l];
     l=index(i+1,j-2,k,2);
     int p1310=mcL[l];
     l=index(i+1,j-2,k,2);
     int p1311=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,1);
     int p1320=mcL[l];
     l=index(i+1,j-1,k+1,1);
     int p1321=mcL[l]*mcL[l];
     l=index(i-2,j+1,k-1,1);
     int p1330=mcL[l];
     l=index(i-2,j+1,k-1,1);
     int p1331=mcL[l]*mcL[l];
     l=index(i+1,j-2,k-1,1);
     int p1340=mcL[l];
     l=index(i+1,j-2,k-1,1);
     int p1341=mcL[l]*mcL[l];
     l=index(i-2,j+1,k+1,1);
     int p1350=mcL[l];
     l=index(i-2,j+1,k+1,1);
     int p1351=mcL[l]*mcL[l];
     l=index(i+1,j-2,k+1,1);
     int p1360=mcL[l];
     l=index(i+1,j-2,k+1,1);
     int p1361=mcL[l]*mcL[l];
     l=index(i+1,j,k-2,2);
     int p1370=mcL[l];
     l=index(i+1,j,k-2,2);
     int p1371=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,1);
     int p1380=mcL[l];
     l=index(i+1,j+1,k-1,1);
     int p1381=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-2,1);
     int p1390=mcL[l];
     l=index(i+1,j+1,k-2,1);
     int p1391=mcL[l]*mcL[l];
     l=index(i,j-2,k-2,2);
     int p1400=mcL[l];
     l=index(i,j-2,k-2,2);
     int p1401=mcL[l]*mcL[l];
     l=index(i,j-2,k+1,2);
     int p1410=mcL[l];
     l=index(i,j-2,k+1,2);
     int p1411=mcL[l]*mcL[l];
     l=index(i,j+1,k-2,2);
     int p1420=mcL[l];
     l=index(i,j+1,k-2,2);
     int p1421=mcL[l]*mcL[l];

     energy = energy-7.46574*(p01)+0.981589*(p00)-0.0173448*(p11*p00+p21*p00+p31*p00+p41*p00+p51*p00+p61*p00+p71*p00+p81*p00)-0.159011*(p10*p01+p20*p01+p30*p01+p40*p01+p50*p01+p60*p01+p70*p01+p80*p01)+0.679524*(p11*p01+p21*p01+p31*p01+p41*p01+p51*p01+p61*p01+p71*p01+p81*p01)+0.0841386*(p10*p00+p20*p00+p30*p00+p40*p00+p50*p00+p60*p00+p70*p00+p80*p00)-0.0349138*(p01*p90+p101*p00+p01*p100+p91*p00+p01*p110+p121*p00+p01*p120+p111*p00+p01*p130+p141*p00+p01*p140+p131*p00+p01*p150+p161*p00+p01*p160+p151*p00+p01*p170+p181*p00+p01*p180+p171*p00+p01*p190+p201*p00+p01*p200+p191*p00)-0.118999*(p01*p210+p221*p00+p01*p220+p211*p00+p01*p230+p241*p00+p01*p240+p231*p00+p01*p250+p261*p00+p01*p260+p251*p00)-0.00161467*(p271*p01+p281*p01+p291*p01+p301*p01+p311*p01+p321*p01+p331*p01+p341*p01+p351*p01+p361*p01+p371*p01+p381*p01+p391*p01+p401*p01+p411*p01+p421*p01+p431*p01+p441*p01+p451*p01+p461*p01+p471*p01+p481*p01+p491*p01+p501*p01)-0.0199544*(p01*p511+p521*p01+p01*p531+p541*p01+p01*p551+p561*p01+p01*p571+p581*p01+p01*p591+p601*p01+p01*p611+p621*p01+p01*p631+p641*p01+p01*p651+p661*p01+p01*p671+p681*p01+p01*p691+p701*p01+p01*p711+p721*p01+p01*p731+p741*p01)-0.00723911*(p00*p510+p520*p00+p00*p530+p540*p00+p00*p550+p560*p00+p00*p570+p580*p00+p00*p590+p600*p00+p00*p610+p620*p00+p00*p630+p640*p00+p00*p650+p660*p00+p00*p670+p680*p00+p00*p690+p700*p00+p00*p710+p720*p00+p00*p730+p740*p00)-0.0383274*(p751*p00+p761*p00+p771*p00+p781*p00+p791*p00+p801*p00+p811*p00+p821*p00)+0.00311609*(p751*p01+p761*p01+p771*p01+p781*p01+p791*p01+p801*p01+p811*p01+p821*p01)+0.0421336*(p750*p00+p760*p00+p770*p00+p780*p00+p790*p00+p800*p00+p810*p00+p820*p00)-0.00852922*(p00*p830+p840*p00+p00*p850+p860*p00+p00*p870+p880*p00+p00*p890+p900*p00+p00*p910+p920*p00+p00*p930+p940*p00)-0.00109746*(p01*p100*p120+p91*p00*p190+p111*p200*p00+p01*p200*p110+p191*p00*p90+p121*p100*p00+p01*p190*p90+p201*p00*p110+p101*p120*p00+p01*p100*p140+p91*p00*p150+p131*p160*p00+p01*p160*p130+p151*p00*p90+p141*p100*p00+p01*p150*p90+p161*p00*p130+p101*p140*p00+p01*p100*p160+p91*p00*p130+p151*p140*p00+p01*p140*p150+p131*p00*p90+p161*p100*p00+p01*p130*p90+p141*p00*p150+p101*p160*p00+p01*p100*p200+p91*p00*p110+p191*p120*p00+p01*p120*p190+p111*p00*p90+p201*p100*p00+p01*p110*p90+p121*p00*p190+p101*p200*p00+p01*p120*p140+p111*p00*p170+p131*p180*p00+p01*p180*p130+p171*p00*p110+p141*p120*p00+p01*p170*p110+p181*p00*p130+p121*p140*p00+p01*p120*p180+p111*p00*p130+p171*p140*p00+p01*p140*p170+p131*p00*p110+p181*p120*p00+p01*p130*p110+p141*p00*p170+p121*p180*p00+p01*p160*p180+p151*p00*p190+p171*p200*p00+p01*p200*p170+p191*p00*p150+p181*p160*p00+p01*p190*p150+p201*p00*p170+p161*p180*p00+p01*p160*p200+p151*p00*p170+p191*p180*p00+p01*p180*p190+p171*p00*p150+p201*p160*p00+p01*p170*p150+p181*p00*p190+p161*p200*p00)+0.0019137*(p00*p100*p120+p90*p00*p190+p110*p200*p00+p00*p100*p140+p90*p00*p150+p130*p160*p00+p00*p100*p160+p90*p00*p130+p150*p140*p00+p00*p100*p200+p90*p00*p110+p190*p120*p00+p00*p120*p140+p110*p00*p170+p130*p180*p00+p00*p120*p180+p110*p00*p130+p170*p140*p00+p00*p160*p180+p150*p00*p190+p170*p200*p00+p00*p160*p200+p150*p00*p170+p190*p180*p00)-0.00450773*(p00*p100*p951+p90*p00*p961+p00*p100*p971+p90*p00*p981+p00*p120*p991+p110*p00*p1001+p00*p120*p1011+p110*p00*p1021+p00*p140*p1031+p130*p00*p1041+p00*p140*p1051+p130*p00*p1061+p00*p160*p1071+p150*p00*p1081+p00*p160*p1091+p150*p00*p1101+p00*p180*p1111+p170*p00*p1121+p00*p180*p1131+p170*p00*p1141+p00*p200*p1151+p190*p00*p1161+p00*p200*p1171+p190*p00*p1181)-0.00212345*(p00*p100*p950+p90*p00*p960+p00*p100*p970+p90*p00*p980+p00*p120*p990+p110*p00*p1000+p00*p120*p1010+p110*p00*p1020+p00*p140*p1030+p130*p00*p1040+p00*p140*p1050+p130*p00*p1060+p00*p160*p1070+p150*p00*p1080+p00*p160*p1090+p150*p00*p1100+p00*p180*p1110+p170*p00*p1120+p00*p180*p1130+p170*p00*p1140+p00*p200*p1150+p190*p00*p1160+p00*p200*p1170+p190*p00*p1180)-0.00158958*(p70*p00*p221+p1170*p210*p01+p70*p00*p241+p1090*p230*p01+p80*p00*p221+p1050*p210*p01+p80*p00*p241+p1010*p230*p01+p60*p00*p221+p1180*p210*p01+p60*p00*p251+p1130*p260*p01+p80*p00*p251+p970*p260*p01+p50*p00*p241+p1100*p230*p01+p50*p00*p251+p1140*p260*p01+p40*p00*p221+p1060*p210*p01+p40*p00*p261+p1110*p250*p01+p70*p00*p261+p950*p250*p01+p40*p00*p231+p1070*p240*p01+p60*p00*p231+p990*p240*p01+p30*p00*p241+p1020*p230*p01+p30*p00*p261+p1120*p250*p01+p20*p00*p251+p980*p260*p01+p20*p00*p231+p1080*p240*p01+p30*p00*p211+p1150*p220*p01+p50*p00*p211+p1030*p220*p01+p20*p00*p211+p1160*p220*p01+p10*p00*p261+p960*p250*p01+p10*p00*p231+p1000*p240*p01+p10*p00*p211+p1040*p220*p01)+0.0012753*(p01*p180*p90+p171*p00*p210+p101*p220*p00+p01*p170*p90+p181*p00*p230+p101*p240*p00+p01*p160*p110+p151*p00*p210+p121*p220*p00+p01*p110*p150+p121*p00*p250+p161*p260*p00+p01*p200*p130+p191*p00*p230+p141*p240*p00+p01*p130*p190+p141*p00*p250+p201*p260*p00+p01*p120*p150+p111*p00*p210+p161*p220*p00+p01*p100*p170+p91*p00*p210+p181*p220*p00+p01*p140*p190+p131*p00*p230+p201*p240*p00+p01*p200*p140+p191*p00*p250+p131*p260*p00+p01*p100*p180+p91*p00*p230+p171*p240*p00+p01*p160*p120+p151*p00*p250+p111*p260*p00)+0.0196953*(p01*p181*p91+p171*p01*p211+p101*p221*p01+p01*p171*p91+p181*p01*p231+p101*p241*p01+p01*p161*p111+p151*p01*p211+p121*p221*p01+p01*p111*p151+p121*p01*p251+p161*p261*p01+p01*p201*p131+p191*p01*p231+p141*p241*p01+p01*p131*p191+p141*p01*p251+p201*p261*p01+p01*p121*p151+p111*p01*p211+p161*p221*p01+p01*p101*p171+p91*p01*p211+p181*p221*p01+p01*p141*p191+p131*p01*p231+p201*p241*p01+p01*p201*p141+p191*p01*p251+p131*p261*p01+p01*p101*p181+p91*p01*p231+p171*p241*p01+p01*p161*p121+p151*p01*p251+p111*p261*p01)+5.94503e-05*(p00*p180*p90+p170*p00*p210+p100*p220*p00+p00*p170*p90+p180*p00*p230+p100*p240*p00+p00*p160*p110+p150*p00*p210+p120*p220*p00+p00*p110*p150+p120*p00*p250+p160*p260*p00+p00*p200*p130+p190*p00*p230+p140*p240*p00+p00*p130*p190+p140*p00*p250+p200*p260*p00+p00*p120*p150+p110*p00*p210+p160*p220*p00+p00*p100*p170+p90*p00*p210+p180*p220*p00+p00*p140*p190+p130*p00*p230+p200*p240*p00+p00*p200*p140+p190*p00*p250+p130*p260*p00+p00*p100*p180+p90*p00*p230+p170*p240*p00+p00*p160*p120+p150*p00*p250+p110*p260*p00)-3.18114e-05*(p01*p100*p471+p91*p00*p381+p01*p90*p381+p101*p00*p471+p01*p100*p451+p91*p00*p341+p01*p90*p341+p101*p00*p451+p01*p100*p431+p91*p00*p321+p01*p90*p321+p101*p00*p431+p01*p100*p391+p91*p00*p301+p01*p90*p301+p101*p00*p391+p01*p120*p491+p111*p00*p401+p01*p110*p401+p121*p00*p491+p01*p120*p451+p111*p00*p361+p01*p110*p361+p121*p00*p451+p01*p120*p411+p111*p00*p321+p01*p110*p321+p121*p00*p411+p01*p120*p371+p111*p00*p281+p01*p110*p281+p121*p00*p371+p01*p140*p491+p131*p00*p441+p01*p130*p441+p141*p00*p491+p01*p140*p471+p131*p00*p421+p01*p130*p421+p141*p00*p471+p01*p140*p351+p131*p00*p301+p01*p130*p301+p141*p00*p351+p01*p140*p331+p131*p00*p281+p01*p130*p281+p141*p00*p331+p01*p160*p501+p151*p00*p461+p01*p150*p461+p161*p00*p501+p01*p160*p421+p151*p00*p381+p01*p150*p381+p161*p00*p421+p01*p160*p391+p151*p00*p351+p01*p150*p351+p161*p00*p391+p01*p160*p311+p151*p00*p271+p01*p150*p271+p161*p00*p311+p01*p180*p481+p171*p00*p461+p01*p170*p461+p181*p00*p481+p01*p180*p441+p171*p00*p401+p01*p170*p401+p181*p00*p441+p01*p180*p371+p171*p00*p331+p01*p170*p331+p181*p00*p371+p01*p180*p311+p171*p00*p291+p01*p170*p291+p181*p00*p311+p01*p200*p501+p191*p00*p481+p01*p190*p481+p201*p00*p501+p01*p200*p431+p191*p00*p411+p01*p190*p411+p201*p00*p431+p01*p200*p361+p191*p00*p341+p01*p190*p341+p201*p00*p361+p01*p200*p291+p191*p00*p271+p01*p190*p271+p201*p00*p291)+0.00127467*(p00*p121*p91+p110*p01*p511+p100*p521*p01+p00*p141*p91+p130*p01*p531+p100*p541*p01+p00*p161*p91+p150*p01*p551+p100*p561*p01+p00*p201*p91+p190*p01*p571+p100*p581*p01+p00*p101*p111+p90*p01*p511+p120*p521*p01+p00*p141*p111+p130*p01*p591+p120*p601*p01+p00*p181*p111+p170*p01*p611+p120*p621*p01+p00*p111*p191+p120*p01*p631+p200*p641*p01+p00*p101*p131+p90*p01*p531+p140*p541*p01+p00*p121*p131+p110*p01*p591+p140*p601*p01+p00*p171*p131+p180*p01*p651+p140*p661*p01+p00*p131*p151+p140*p01*p671+p160*p681*p01+p00*p101*p151+p90*p01*p551+p160*p561*p01+p00*p181*p151+p170*p01*p691+p160*p701*p01+p00*p201*p151+p190*p01*p711+p160*p721*p01+p00*p121*p171+p110*p01*p611+p180*p621*p01+p00*p161*p171+p150*p01*p691+p180*p701*p01+p00*p191*p171+p200*p01*p731+p180*p741*p01+p00*p101*p191+p90*p01*p571+p200*p581*p01+p00*p161*p191+p150*p01*p711+p200*p721*p01+p00*p201*p121+p190*p01*p631+p110*p641*p01+p00*p181*p201+p170*p01*p731+p190*p741*p01+p00*p141*p181+p130*p01*p651+p170*p661*p01+p00*p161*p141+p150*p01*p671+p130*p681*p01)-0.000997247*(p00*p120*p90+p110*p00*p510+p100*p520*p00+p00*p140*p90+p130*p00*p530+p100*p540*p00+p00*p160*p90+p150*p00*p550+p100*p560*p00+p00*p200*p90+p190*p00*p570+p100*p580*p00+p00*p100*p110+p90*p00*p510+p120*p520*p00+p00*p140*p110+p130*p00*p590+p120*p600*p00+p00*p180*p110+p170*p00*p610+p120*p620*p00+p00*p110*p190+p120*p00*p630+p200*p640*p00+p00*p100*p130+p90*p00*p530+p140*p540*p00+p00*p120*p130+p110*p00*p590+p140*p600*p00+p00*p170*p130+p180*p00*p650+p140*p660*p00+p00*p130*p150+p140*p00*p670+p160*p680*p00+p00*p100*p150+p90*p00*p550+p160*p560*p00+p00*p180*p150+p170*p00*p690+p160*p700*p00+p00*p200*p150+p190*p00*p710+p160*p720*p00+p00*p120*p170+p110*p00*p610+p180*p620*p00+p00*p160*p170+p150*p00*p690+p180*p700*p00+p00*p190*p170+p200*p00*p730+p180*p740*p00+p00*p100*p190+p90*p00*p570+p200*p580*p00+p00*p160*p190+p150*p00*p710+p200*p720*p00+p00*p200*p120+p190*p00*p630+p110*p640*p00+p00*p180*p200+p170*p00*p730+p190*p740*p00+p00*p140*p180+p130*p00*p650+p170*p660*p00+p00*p160*p140+p150*p00*p670+p130*p680*p00)-0.00090179*(p01*p90*p250+p101*p00*p600+p261*p590*p00+p01*p90*p260+p101*p00*p720+p251*p710*p00+p01*p110*p240+p121*p00*p540+p231*p530*p00+p01*p110*p230+p121*p00*p740+p241*p730*p00+p01*p130*p220+p141*p00*p520+p211*p510*p00+p01*p130*p210+p141*p00*p690+p221*p700*p00+p01*p150*p240+p161*p00*p580+p231*p570*p00+p01*p150*p230+p161*p00*p650+p241*p660*p00+p01*p170*p250+p181*p00*p630+p261*p640*p00+p01*p170*p260+p181*p00*p680+p251*p670*p00+p01*p190*p220+p201*p00*p560+p211*p550*p00+p01*p190*p210+p201*p00*p610+p221*p620*p00+p01*p200*p220+p191*p00*p620+p211*p610*p00+p01*p200*p210+p191*p00*p550+p221*p560*p00+p01*p180*p250+p171*p00*p670+p261*p680*p00+p01*p180*p260+p171*p00*p640+p251*p630*p00+p01*p160*p240+p151*p00*p660+p231*p650*p00+p01*p160*p230+p151*p00*p570+p241*p580*p00+p01*p140*p220+p131*p00*p700+p211*p690*p00+p01*p140*p210+p131*p00*p510+p221*p520*p00+p01*p120*p240+p111*p00*p730+p231*p740*p00+p01*p120*p230+p111*p00*p530+p241*p540*p00+p01*p100*p250+p91*p00*p710+p261*p720*p00+p01*p100*p260+p91*p00*p590+p251*p600*p00)+0.00599458*(p00*p90*p251+p100*p00*p601+p260*p590*p01+p00*p90*p261+p100*p00*p721+p250*p710*p01+p00*p110*p241+p120*p00*p541+p230*p530*p01+p00*p110*p231+p120*p00*p741+p240*p730*p01+p00*p130*p221+p140*p00*p521+p210*p510*p01+p00*p130*p211+p140*p00*p691+p220*p700*p01+p00*p150*p241+p160*p00*p581+p230*p570*p01+p00*p150*p231+p160*p00*p651+p240*p660*p01+p00*p170*p251+p180*p00*p631+p260*p640*p01+p00*p170*p261+p180*p00*p681+p250*p670*p01+p00*p190*p221+p200*p00*p561+p210*p550*p01+p00*p190*p211+p200*p00*p611+p220*p620*p01+p00*p200*p221+p190*p00*p621+p210*p610*p01+p00*p200*p211+p190*p00*p551+p220*p560*p01+p00*p180*p251+p170*p00*p671+p260*p680*p01+p00*p180*p261+p170*p00*p641+p250*p630*p01+p00*p160*p241+p150*p00*p661+p230*p650*p01+p00*p160*p231+p150*p00*p571+p240*p580*p01+p00*p140*p221+p130*p00*p701+p210*p690*p01+p00*p140*p211+p130*p00*p511+p220*p520*p01+p00*p120*p241+p110*p00*p731+p230*p740*p01+p00*p120*p231+p110*p00*p531+p240*p540*p01+p00*p100*p251+p90*p00*p711+p260*p720*p01+p00*p100*p261+p90*p00*p591+p250*p600*p01)+0.000252238*(p00*p90*p250+p100*p00*p600+p260*p590*p00+p00*p90*p260+p100*p00*p720+p250*p710*p00+p00*p110*p240+p120*p00*p540+p230*p530*p00+p00*p110*p230+p120*p00*p740+p240*p730*p00+p00*p130*p220+p140*p00*p520+p210*p510*p00+p00*p130*p210+p140*p00*p690+p220*p700*p00+p00*p150*p240+p160*p00*p580+p230*p570*p00+p00*p150*p230+p160*p00*p650+p240*p660*p00+p00*p170*p250+p180*p00*p630+p260*p640*p00+p00*p170*p260+p180*p00*p680+p250*p670*p00+p00*p190*p220+p200*p00*p560+p210*p550*p00+p00*p190*p210+p200*p00*p610+p220*p620*p00+p00*p200*p220+p190*p00*p620+p210*p610*p00+p00*p200*p210+p190*p00*p550+p220*p560*p00+p00*p180*p250+p170*p00*p670+p260*p680*p00+p00*p180*p260+p170*p00*p640+p250*p630*p00+p00*p160*p240+p150*p00*p660+p230*p650*p00+p00*p160*p230+p150*p00*p570+p240*p580*p00+p00*p140*p220+p130*p00*p700+p210*p690*p00+p00*p140*p210+p130*p00*p510+p220*p520*p00+p00*p120*p240+p110*p00*p730+p230*p740*p00+p00*p120*p230+p110*p00*p530+p240*p540*p00+p00*p100*p250+p90*p00*p710+p260*p720*p00+p00*p100*p260+p90*p00*p590+p250*p600*p00)+0.00780764*(p1030*p01*p700+p450*p691*p00+p990*p01*p730+p470*p741*p00+p1150*p01*p620+p390*p611*p00+p1070*p01*p660+p430*p651*p00+p950*p01*p710+p490*p721*p00+p1160*p01*p560+p370*p551*p00+p1110*p01*p670+p410*p681*p00+p1080*p01*p580+p330*p571*p00+p1120*p01*p630+p350*p641*p00+p1040*p01*p520+p310*p511*p00+p970*p01*p590+p500*p601*p00+p1130*p01*p640+p420*p631*p00+p1010*p01*p530+p480*p541*p00+p1090*p01*p570+p440*p581*p00+p1000*p01*p540+p290*p531*p00+p1140*p01*p680+p360*p671*p00+p960*p01*p600+p270*p591*p00+p1100*p01*p650+p340*p661*p00+p1050*p01*p510+p460*p521*p00+p1170*p01*p550+p400*p561*p00+p1180*p01*p610+p380*p621*p00+p980*p01*p720+p280*p711*p00+p1020*p01*p740+p300*p731*p00+p1060*p01*p690+p320*p701*p00)-0.000457364*(p71*p00*p520+p291*p510*p00+p71*p00*p540+p311*p530*p00+p81*p00*p560+p331*p550*p00+p81*p00*p580+p371*p570*p00+p61*p00*p520+p271*p510*p00+p61*p00*p600+p311*p590*p00+p81*p00*p620+p351*p610*p00+p81*p00*p630+p391*p640*p00+p51*p00*p540+p271*p530*p00+p51*p00*p600+p291*p590*p00+p81*p00*p660+p411*p650*p00+p81*p00*p670+p431*p680*p00+p41*p00*p560+p281*p550*p00+p71*p00*p700+p361*p690*p00+p41*p00*p720+p371*p710*p00+p71*p00*p680+p451*p670*p00+p41*p00*p620+p301*p610*p00+p61*p00*p700+p341*p690*p00+p41*p00*p740+p391*p730*p00+p61*p00*p650+p451*p660*p00+p31*p00*p580+p281*p570*p00+p31*p00*p720+p331*p710*p00+p71*p00*p730+p421*p740*p00+p71*p00*p640+p471*p630*p00+p21*p00*p630+p301*p640*p00+p21*p00*p740+p351*p730*p00+p61*p00*p710+p441*p720*p00+p61*p00*p570+p491*p580*p00+p31*p00*p660+p321*p650*p00+p51*p00*p730+p381*p740*p00+p31*p00*p690+p431*p700*p00+p51*p00*p610+p471*p620*p00+p21*p00*p670+p321*p680*p00+p51*p00*p710+p401*p720*p00+p21*p00*p690+p411*p700*p00+p51*p00*p550+p491*p560*p00+p11*p00*p680+p341*p670*p00+p11*p00*p650+p361*p660*p00+p41*p00*p590+p481*p600*p00+p41*p00*p530+p501*p540*p00+p11*p00*p640+p381*p630*p00+p11*p00*p610+p421*p620*p00+p31*p00*p590+p461*p600*p00+p31*p00*p510+p501*p520*p00+p11*p00*p570+p401*p580*p00+p11*p00*p550+p441*p560*p00+p21*p00*p530+p461*p540*p00+p21*p00*p510+p481*p520*p00)+0.00472893*(p01*p101*p620+p91*p01*p740+p611*p731*p00+p01*p101*p660+p91*p01*p690+p651*p701*p00+p01*p101*p700+p91*p01*p650+p691*p661*p00+p01*p101*p730+p91*p01*p610+p741*p621*p00+p01*p121*p560+p111*p01*p720+p551*p711*p00+p01*p121*p670+p111*p01*p690+p681*p701*p00+p01*p121*p700+p111*p01*p680+p691*p671*p00+p01*p121*p710+p111*p01*p550+p721*p561*p00+p01*p141*p580+p131*p01*p720+p571*p711*p00+p01*p141*p630+p131*p01*p740+p641*p731*p00+p01*p141*p730+p131*p01*p640+p741*p631*p00+p01*p141*p710+p131*p01*p570+p721*p581*p00+p01*p161*p520+p151*p01*p600+p511*p591*p00+p01*p161*p620+p151*p01*p630+p611*p641*p00+p01*p161*p640+p151*p01*p610+p631*p621*p00+p01*p161*p590+p151*p01*p510+p601*p521*p00+p01*p181*p520+p171*p01*p540+p511*p531*p00+p01*p181*p560+p171*p01*p580+p551*p571*p00+p01*p181*p570+p171*p01*p550+p581*p561*p00+p01*p181*p530+p171*p01*p510+p541*p521*p00+p01*p201*p540+p191*p01*p600+p531*p591*p00+p01*p201*p680+p191*p01*p650+p671*p661*p00+p01*p201*p660+p191*p01*p670+p651*p681*p00+p01*p201*p590+p191*p01*p530+p601*p541*p00)-0.00191757*(p00*p100*p621+p90*p00*p741+p610*p730*p01+p00*p100*p661+p90*p00*p691+p650*p700*p01+p00*p100*p701+p90*p00*p651+p690*p660*p01+p00*p100*p731+p90*p00*p611+p740*p620*p01+p00*p120*p561+p110*p00*p721+p550*p710*p01+p00*p120*p671+p110*p00*p691+p680*p700*p01+p00*p120*p701+p110*p00*p681+p690*p670*p01+p00*p120*p711+p110*p00*p551+p720*p560*p01+p00*p140*p581+p130*p00*p721+p570*p710*p01+p00*p140*p631+p130*p00*p741+p640*p730*p01+p00*p140*p731+p130*p00*p641+p740*p630*p01+p00*p140*p711+p130*p00*p571+p720*p580*p01+p00*p160*p521+p150*p00*p601+p510*p590*p01+p00*p160*p621+p150*p00*p631+p610*p640*p01+p00*p160*p641+p150*p00*p611+p630*p620*p01+p00*p160*p591+p150*p00*p511+p600*p520*p01+p00*p180*p521+p170*p00*p541+p510*p530*p01+p00*p180*p561+p170*p00*p581+p550*p570*p01+p00*p180*p571+p170*p00*p551+p580*p560*p01+p00*p180*p531+p170*p00*p511+p540*p520*p01+p00*p200*p541+p190*p00*p601+p530*p590*p01+p00*p200*p681+p190*p00*p651+p670*p660*p01+p00*p200*p661+p190*p00*p671+p650*p680*p01+p00*p200*p591+p190*p00*p531+p600*p540*p01)+0.00942619*(p01*p101*p621+p91*p01*p741+p611*p731*p01+p01*p101*p661+p91*p01*p691+p651*p701*p01+p01*p101*p701+p91*p01*p651+p691*p661*p01+p01*p101*p731+p91*p01*p611+p741*p621*p01+p01*p121*p561+p111*p01*p721+p551*p711*p01+p01*p121*p671+p111*p01*p691+p681*p701*p01+p01*p121*p701+p111*p01*p681+p691*p671*p01+p01*p121*p711+p111*p01*p551+p721*p561*p01+p01*p141*p581+p131*p01*p721+p571*p711*p01+p01*p141*p631+p131*p01*p741+p641*p731*p01+p01*p141*p731+p131*p01*p641+p741*p631*p01+p01*p141*p711+p131*p01*p571+p721*p581*p01+p01*p161*p521+p151*p01*p601+p511*p591*p01+p01*p161*p621+p151*p01*p631+p611*p641*p01+p01*p161*p641+p151*p01*p611+p631*p621*p01+p01*p161*p591+p151*p01*p511+p601*p521*p01+p01*p181*p521+p171*p01*p541+p511*p531*p01+p01*p181*p561+p171*p01*p581+p551*p571*p01+p01*p181*p571+p171*p01*p551+p581*p561*p01+p01*p181*p531+p171*p01*p511+p541*p521*p01+p01*p201*p541+p191*p01*p601+p531*p591*p01+p01*p201*p681+p191*p01*p651+p671*p661*p01+p01*p201*p661+p191*p01*p671+p651*p681*p01+p01*p201*p591+p191*p01*p531+p601*p541*p01)-0.00108305*(p00*p100*p620+p90*p00*p740+p610*p730*p00+p00*p100*p660+p90*p00*p690+p650*p700*p00+p00*p100*p700+p90*p00*p650+p690*p660*p00+p00*p100*p730+p90*p00*p610+p740*p620*p00+p00*p120*p560+p110*p00*p720+p550*p710*p00+p00*p120*p670+p110*p00*p690+p680*p700*p00+p00*p120*p700+p110*p00*p680+p690*p670*p00+p00*p120*p710+p110*p00*p550+p720*p560*p00+p00*p140*p580+p130*p00*p720+p570*p710*p00+p00*p140*p630+p130*p00*p740+p640*p730*p00+p00*p140*p730+p130*p00*p640+p740*p630*p00+p00*p140*p710+p130*p00*p570+p720*p580*p00+p00*p160*p520+p150*p00*p600+p510*p590*p00+p00*p160*p620+p150*p00*p630+p610*p640*p00+p00*p160*p640+p150*p00*p610+p630*p620*p00+p00*p160*p590+p150*p00*p510+p600*p520*p00+p00*p180*p520+p170*p00*p540+p510*p530*p00+p00*p180*p560+p170*p00*p580+p550*p570*p00+p00*p180*p570+p170*p00*p550+p580*p560*p00+p00*p180*p530+p170*p00*p510+p540*p520*p00+p00*p200*p540+p190*p00*p600+p530*p590*p00+p00*p200*p680+p190*p00*p650+p670*p660*p00+p00*p200*p660+p190*p00*p670+p650*p680*p00+p00*p200*p590+p190*p00*p530+p600*p540*p00)+0.00424329*(p00*p220*p541+p210*p00*p661+p530*p650*p01+p00*p240*p521+p230*p00*p621+p510*p610*p01+p00*p220*p581+p210*p00*p731+p570*p740*p01+p00*p240*p561+p230*p00*p701+p550*p690*p01+p00*p220*p601+p210*p00*p671+p590*p680*p01+p00*p260*p561+p250*p00*p521+p550*p510*p01+p00*p220*p631+p210*p00*p711+p640*p720*p01+p00*p260*p701+p250*p00*p621+p690*p610*p01+p00*p240*p601+p230*p00*p631+p590*p640*p01+p00*p260*p581+p250*p00*p541+p570*p530*p01+p00*p240*p671+p230*p00*p711+p680*p720*p01+p00*p260*p731+p250*p00*p661+p740*p650*p01+p00*p220*p721+p210*p00*p641+p710*p630*p01+p00*p220*p681+p210*p00*p591+p670*p600*p01+p00*p220*p741+p210*p00*p571+p730*p580*p01+p00*p220*p651+p210*p00*p531+p660*p540*p01+p00*p240*p721+p230*p00*p681+p710*p670*p01+p00*p240*p641+p230*p00*p591+p630*p600*p01+p00*p260*p651+p250*p00*p741+p660*p730*p01+p00*p260*p531+p250*p00*p571+p540*p580*p01+p00*p240*p691+p230*p00*p551+p700*p560*p01+p00*p240*p611+p230*p00*p511+p620*p520*p01+p00*p260*p611+p250*p00*p691+p620*p700*p01+p00*p260*p511+p250*p00*p551+p520*p560*p01)+0.00615635*(p01*p220*p541+p211*p00*p661+p531*p650*p01+p01*p210*p661+p221*p00*p541+p651*p530*p01+p01*p240*p521+p231*p00*p621+p511*p610*p01+p01*p230*p621+p241*p00*p521+p611*p510*p01+p01*p220*p581+p211*p00*p731+p571*p740*p01+p01*p210*p731+p221*p00*p581+p741*p570*p01+p01*p240*p561+p231*p00*p701+p551*p690*p01+p01*p230*p701+p241*p00*p561+p691*p550*p01+p01*p220*p601+p211*p00*p671+p591*p680*p01+p01*p210*p671+p221*p00*p601+p681*p590*p01+p01*p260*p561+p251*p00*p521+p551*p510*p01+p01*p250*p521+p261*p00*p561+p511*p550*p01+p01*p220*p631+p211*p00*p711+p641*p720*p01+p01*p210*p711+p221*p00*p631+p721*p640*p01+p01*p260*p701+p251*p00*p621+p691*p610*p01+p01*p250*p621+p261*p00*p701+p611*p690*p01+p01*p240*p601+p231*p00*p631+p591*p640*p01+p01*p230*p631+p241*p00*p601+p641*p590*p01+p01*p260*p581+p251*p00*p541+p571*p530*p01+p01*p250*p541+p261*p00*p581+p531*p570*p01+p01*p240*p671+p231*p00*p711+p681*p720*p01+p01*p230*p711+p241*p00*p671+p721*p680*p01+p01*p260*p731+p251*p00*p661+p741*p650*p01+p01*p250*p661+p261*p00*p731+p651*p740*p01+p01*p220*p721+p211*p00*p641+p711*p630*p01+p01*p210*p641+p221*p00*p721+p631*p710*p01+p01*p220*p681+p211*p00*p591+p671*p600*p01+p01*p210*p591+p221*p00*p681+p601*p670*p01+p01*p220*p741+p211*p00*p571+p731*p580*p01+p01*p210*p571+p221*p00*p741+p581*p730*p01+p01*p220*p651+p211*p00*p531+p661*p540*p01+p01*p210*p531+p221*p00*p651+p541*p660*p01+p01*p240*p721+p231*p00*p681+p711*p670*p01+p01*p230*p681+p241*p00*p721+p671*p710*p01+p01*p240*p641+p231*p00*p591+p631*p600*p01+p01*p230*p591+p241*p00*p641+p601*p630*p01+p01*p260*p651+p251*p00*p741+p661*p730*p01+p01*p250*p741+p261*p00*p651+p731*p660*p01+p01*p260*p531+p251*p00*p571+p541*p580*p01+p01*p250*p571+p261*p00*p531+p581*p540*p01+p01*p240*p691+p231*p00*p551+p701*p560*p01+p01*p230*p551+p241*p00*p691+p561*p700*p01+p01*p240*p611+p231*p00*p511+p621*p520*p01+p01*p230*p511+p241*p00*p611+p521*p620*p01+p01*p260*p611+p251*p00*p691+p621*p700*p01+p01*p250*p691+p261*p00*p611+p701*p620*p01+p01*p260*p511+p251*p00*p551+p521*p560*p01+p01*p250*p551+p261*p00*p511+p561*p520*p01)+6.45141e-05*(p00*p220*p540+p210*p00*p660+p530*p650*p00+p00*p240*p520+p230*p00*p620+p510*p610*p00+p00*p220*p580+p210*p00*p730+p570*p740*p00+p00*p240*p560+p230*p00*p700+p550*p690*p00+p00*p220*p600+p210*p00*p670+p590*p680*p00+p00*p260*p560+p250*p00*p520+p550*p510*p00+p00*p220*p630+p210*p00*p710+p640*p720*p00+p00*p260*p700+p250*p00*p620+p690*p610*p00+p00*p240*p600+p230*p00*p630+p590*p640*p00+p00*p260*p580+p250*p00*p540+p570*p530*p00+p00*p240*p670+p230*p00*p710+p680*p720*p00+p00*p260*p730+p250*p00*p660+p740*p650*p00+p00*p220*p720+p210*p00*p640+p710*p630*p00+p00*p220*p680+p210*p00*p590+p670*p600*p00+p00*p220*p740+p210*p00*p570+p730*p580*p00+p00*p220*p650+p210*p00*p530+p660*p540*p00+p00*p240*p720+p230*p00*p680+p710*p670*p00+p00*p240*p640+p230*p00*p590+p630*p600*p00+p00*p260*p650+p250*p00*p740+p660*p730*p00+p00*p260*p530+p250*p00*p570+p540*p580*p00+p00*p240*p690+p230*p00*p550+p700*p560*p00+p00*p240*p610+p230*p00*p510+p620*p520*p00+p00*p260*p610+p250*p00*p690+p620*p700*p00+p00*p260*p510+p250*p00*p550+p520*p560*p00)-0.00480924*(p01*p540*p560+p531*p00*p680+p551*p670*p00+p01*p670*p550+p681*p00*p530+p561*p540*p00+p01*p680*p530+p671*p00*p550+p541*p560*p00+p01*p520*p580+p511*p00*p640+p571*p630*p00+p01*p630*p570+p641*p00*p510+p581*p520*p00+p01*p640*p510+p631*p00*p570+p521*p580*p00+p01*p660*p610+p651*p00*p590+p621*p600*p00+p01*p620*p600+p611*p00*p660+p591*p650*p00+p01*p590*p650+p601*p00*p620+p661*p610*p00+p01*p570*p510+p581*p00*p640+p521*p630*p00+p01*p580*p640+p571*p00*p510+p631*p520*p00+p01*p520*p630+p511*p00*p570+p641*p580*p00+p01*p620*p650+p611*p00*p590+p661*p600*p00+p01*p590*p610+p601*p00*p660+p621*p650*p00+p01*p660*p600+p651*p00*p620+p591*p610*p00+p01*p550*p530+p561*p00*p680+p541*p670*p00+p01*p560*p680+p551*p00*p530+p671*p540*p00+p01*p540*p670+p531*p00*p550+p681*p560*p00+p01*p730*p690+p741*p00*p710+p701*p720*p00+p01*p700*p720+p691*p00*p730+p711*p740*p00+p01*p710*p740+p721*p00*p700+p731*p690*p00+p01*p730*p720+p741*p00*p700+p711*p690*p00+p01*p700*p740+p691*p00*p710+p731*p720*p00+p01*p710*p690+p721*p00*p730+p701*p740*p00)-0.00120959*(p01*p541*p561+p531*p01*p681+p551*p671*p01+p01*p521*p581+p511*p01*p641+p571*p631*p01+p01*p661*p611+p651*p01*p591+p621*p601*p01+p01*p571*p511+p581*p01*p641+p521*p631*p01+p01*p621*p651+p611*p01*p591+p661*p601*p01+p01*p551*p531+p561*p01*p681+p541*p671*p01+p01*p731*p691+p741*p01*p711+p701*p721*p01+p01*p731*p721+p741*p01*p701+p711*p691*p01)+0.00433732*(p1031*p00*p1190+p991*p00*p1200+p1151*p00*p1210+p1071*p00*p1220+p951*p00*p1230+p1161*p00*p1240+p1111*p00*p1250+p1081*p00*p1260+p1121*p00*p1270+p1041*p00*p1280+p971*p00*p1290+p1131*p00*p1300+p1011*p00*p1310+p1091*p00*p1320+p1001*p00*p1330+p1141*p00*p1340+p961*p00*p1350+p1101*p00*p1360+p1051*p00*p1370+p1171*p00*p1380+p1181*p00*p1390+p981*p00*p1400+p1021*p00*p1410+p1061*p00*p1420)+0.00518956*(p1030*p01*p1190+p990*p01*p1200+p1150*p01*p1210+p1070*p01*p1220+p950*p01*p1230+p1160*p01*p1240+p1110*p01*p1250+p1080*p01*p1260+p1120*p01*p1270+p1040*p01*p1280+p970*p01*p1290+p1130*p01*p1300+p1010*p01*p1310+p1090*p01*p1320+p1000*p01*p1330+p1140*p01*p1340+p960*p01*p1350+p1100*p01*p1360+p1050*p01*p1370+p1170*p01*p1380+p1180*p01*p1390+p980*p01*p1400+p1020*p01*p1410+p1060*p01*p1420)+0.0118966*(p1030*p00*p1191+p990*p00*p1201+p1150*p00*p1211+p1070*p00*p1221+p950*p00*p1231+p1160*p00*p1241+p1110*p00*p1251+p1080*p00*p1261+p1120*p00*p1271+p1040*p00*p1281+p970*p00*p1291+p1130*p00*p1301+p1010*p00*p1311+p1090*p00*p1321+p1000*p00*p1331+p1140*p00*p1341+p960*p00*p1351+p1100*p00*p1361+p1050*p00*p1371+p1170*p00*p1381+p1180*p00*p1391+p980*p00*p1401+p1020*p00*p1411+p1060*p00*p1421)+0.00789307*(p01*p90*p420+p101*p00*p1210+p01*p90*p360+p101*p00*p1220+p01*p90*p410+p101*p00*p1190+p01*p90*p350+p101*p00*p1200+p01*p110*p440+p121*p00*p1240+p01*p110*p340+p121*p00*p1250+p01*p110*p430+p121*p00*p1190+p01*p110*p330+p121*p00*p1230+p01*p130*p400+p141*p00*p1260+p01*p130*p380+p141*p00*p1270+p01*p130*p390+p141*p00*p1200+p01*p130*p370+p141*p00*p1230+p01*p150*p480+p161*p00*p1280+p01*p150*p470+p161*p00*p1210+p01*p150*p300+p161*p00*p1300+p01*p150*p290+p161*p00*p1290+p01*p170*p500+p181*p00*p1280+p01*p170*p490+p181*p00*p1240+p01*p170*p280+p181*p00*p1320+p01*p170*p270+p181*p00*p1310+p01*p190*p460+p201*p00*p1330+p01*p190*p320+p201*p00*p1340+p01*p190*p450+p201*p00*p1220+p01*p190*p310+p201*p00*p1290+p01*p200*p460+p191*p00*p1350+p01*p200*p320+p191*p00*p1360+p01*p200*p450+p191*p00*p1250+p01*p200*p310+p191*p00*p1310+p01*p180*p500+p171*p00*p1330+p01*p180*p490+p171*p00*p1260+p01*p180*p280+p171*p00*p1380+p01*p180*p270+p171*p00*p1370+p01*p160*p480+p151*p00*p1350+p01*p160*p470+p151*p00*p1270+p01*p160*p300+p151*p00*p1390+p01*p160*p290+p151*p00*p1370+p01*p140*p400+p131*p00*p1400+p01*p140*p380+p131*p00*p1410+p01*p140*p390+p131*p00*p1300+p01*p140*p370+p131*p00*p1320+p01*p120*p440+p111*p00*p1400+p01*p120*p340+p111*p00*p1420+p01*p120*p430+p111*p00*p1340+p01*p120*p330+p111*p00*p1380+p01*p100*p420+p91*p00*p1410+p01*p100*p360+p91*p00*p1420+p01*p100*p410+p91*p00*p1360+p01*p100*p350+p91*p00*p1390)+0.0011537*(p00*p91*p420+p100*p01*p1210+p00*p91*p360+p100*p01*p1220+p00*p91*p410+p100*p01*p1190+p00*p91*p350+p100*p01*p1200+p00*p111*p440+p120*p01*p1240+p00*p111*p340+p120*p01*p1250+p00*p111*p430+p120*p01*p1190+p00*p111*p330+p120*p01*p1230+p00*p131*p400+p140*p01*p1260+p00*p131*p380+p140*p01*p1270+p00*p131*p390+p140*p01*p1200+p00*p131*p370+p140*p01*p1230+p00*p151*p480+p160*p01*p1280+p00*p151*p470+p160*p01*p1210+p00*p151*p300+p160*p01*p1300+p00*p151*p290+p160*p01*p1290+p00*p171*p500+p180*p01*p1280+p00*p171*p490+p180*p01*p1240+p00*p171*p280+p180*p01*p1320+p00*p171*p270+p180*p01*p1310+p00*p191*p460+p200*p01*p1330+p00*p191*p320+p200*p01*p1340+p00*p191*p450+p200*p01*p1220+p00*p191*p310+p200*p01*p1290+p00*p201*p460+p190*p01*p1350+p00*p201*p320+p190*p01*p1360+p00*p201*p450+p190*p01*p1250+p00*p201*p310+p190*p01*p1310+p00*p181*p500+p170*p01*p1330+p00*p181*p490+p170*p01*p1260+p00*p181*p280+p170*p01*p1380+p00*p181*p270+p170*p01*p1370+p00*p161*p480+p150*p01*p1350+p00*p161*p470+p150*p01*p1270+p00*p161*p300+p150*p01*p1390+p00*p161*p290+p150*p01*p1370+p00*p141*p400+p130*p01*p1400+p00*p141*p380+p130*p01*p1410+p00*p141*p390+p130*p01*p1300+p00*p141*p370+p130*p01*p1320+p00*p121*p440+p110*p01*p1400+p00*p121*p340+p110*p01*p1420+p00*p121*p430+p110*p01*p1340+p00*p121*p330+p110*p01*p1380+p00*p101*p420+p90*p01*p1410+p00*p101*p360+p90*p01*p1420+p00*p101*p410+p90*p01*p1360+p00*p101*p350+p90*p01*p1390)-0.000228738*(p01*p240*p210+p231*p00*p830+p221*p840*p00+p01*p220*p230+p211*p00*p830+p241*p840*p00+p01*p250*p210+p261*p00*p850+p221*p860*p00+p01*p220*p260+p211*p00*p850+p251*p860*p00+p01*p250*p230+p261*p00*p870+p241*p880*p00+p01*p240*p260+p231*p00*p870+p251*p880*p00+p01*p260*p210+p251*p00*p890+p221*p900*p00+p01*p220*p250+p211*p00*p890+p261*p900*p00+p01*p230*p210+p241*p00*p910+p221*p920*p00+p01*p220*p240+p211*p00*p910+p231*p920*p00+p01*p260*p230+p251*p00*p930+p241*p940*p00+p01*p240*p250+p231*p00*p930+p261*p940*p00)-0.00395963*(p00*p241*p210+p230*p01*p830+p220*p841*p00+p00*p211*p240+p220*p01*p840+p230*p831*p00+p00*p221*p230+p210*p01*p830+p240*p841*p00+p00*p231*p220+p240*p01*p840+p210*p831*p00+p00*p251*p210+p260*p01*p850+p220*p861*p00+p00*p211*p250+p220*p01*p860+p260*p851*p00+p00*p221*p260+p210*p01*p850+p250*p861*p00+p00*p261*p220+p250*p01*p860+p210*p851*p00+p00*p251*p230+p260*p01*p870+p240*p881*p00+p00*p231*p250+p240*p01*p880+p260*p871*p00+p00*p241*p260+p230*p01*p870+p250*p881*p00+p00*p261*p240+p250*p01*p880+p230*p871*p00+p00*p261*p210+p250*p01*p890+p220*p901*p00+p00*p211*p260+p220*p01*p900+p250*p891*p00+p00*p221*p250+p210*p01*p890+p260*p901*p00+p00*p251*p220+p260*p01*p900+p210*p891*p00+p00*p231*p210+p240*p01*p910+p220*p921*p00+p00*p211*p230+p220*p01*p920+p240*p911*p00+p00*p221*p240+p210*p01*p910+p230*p921*p00+p00*p241*p220+p230*p01*p920+p210*p911*p00+p00*p261*p230+p250*p01*p930+p240*p941*p00+p00*p231*p260+p240*p01*p940+p250*p931*p00+p00*p241*p250+p230*p01*p930+p260*p941*p00+p00*p251*p240+p260*p01*p940+p230*p931*p00)+0.00530138*(p01*p241*p210+p231*p01*p830+p221*p841*p00+p01*p211*p240+p221*p01*p840+p231*p831*p00+p01*p221*p230+p211*p01*p830+p241*p841*p00+p01*p231*p220+p241*p01*p840+p211*p831*p00+p01*p251*p210+p261*p01*p850+p221*p861*p00+p01*p211*p250+p221*p01*p860+p261*p851*p00+p01*p221*p260+p211*p01*p850+p251*p861*p00+p01*p261*p220+p251*p01*p860+p211*p851*p00+p01*p251*p230+p261*p01*p870+p241*p881*p00+p01*p231*p250+p241*p01*p880+p261*p871*p00+p01*p241*p260+p231*p01*p870+p251*p881*p00+p01*p261*p240+p251*p01*p880+p231*p871*p00+p01*p261*p210+p251*p01*p890+p221*p901*p00+p01*p211*p260+p221*p01*p900+p251*p891*p00+p01*p221*p250+p211*p01*p890+p261*p901*p00+p01*p251*p220+p261*p01*p900+p211*p891*p00+p01*p231*p210+p241*p01*p910+p221*p921*p00+p01*p211*p230+p221*p01*p920+p241*p911*p00+p01*p221*p240+p211*p01*p910+p231*p921*p00+p01*p241*p220+p231*p01*p920+p211*p911*p00+p01*p261*p230+p251*p01*p930+p241*p941*p00+p01*p231*p260+p241*p01*p940+p251*p931*p00+p01*p241*p250+p231*p01*p930+p261*p941*p00+p01*p251*p240+p261*p01*p940+p231*p931*p00)-0.00323454*(p01*p241*p211+p231*p01*p831+p221*p841*p01+p01*p221*p231+p211*p01*p831+p241*p841*p01+p01*p251*p211+p261*p01*p851+p221*p861*p01+p01*p221*p261+p211*p01*p851+p251*p861*p01+p01*p251*p231+p261*p01*p871+p241*p881*p01+p01*p241*p261+p231*p01*p871+p251*p881*p01+p01*p261*p211+p251*p01*p891+p221*p901*p01+p01*p221*p251+p211*p01*p891+p261*p901*p01+p01*p231*p211+p241*p01*p911+p221*p921*p01+p01*p221*p241+p211*p01*p911+p231*p921*p01+p01*p261*p231+p251*p01*p931+p241*p941*p01+p01*p241*p251+p231*p01*p931+p261*p941*p01)-0.000736564*(p00*p240*p210+p230*p00*p830+p220*p840*p00+p00*p220*p230+p210*p00*p830+p240*p840*p00+p00*p250*p210+p260*p00*p850+p220*p860*p00+p00*p220*p260+p210*p00*p850+p250*p860*p00+p00*p250*p230+p260*p00*p870+p240*p880*p00+p00*p240*p260+p230*p00*p870+p250*p880*p00+p00*p260*p210+p250*p00*p890+p220*p900*p00+p00*p220*p250+p210*p00*p890+p260*p900*p00+p00*p230*p210+p240*p00*p910+p220*p920*p00+p00*p220*p240+p210*p00*p910+p230*p920*p00+p00*p260*p230+p250*p00*p930+p240*p940*p00+p00*p240*p250+p230*p00*p930+p260*p940*p00)-0.00891487*(p01*p101*p90+p91*p01*p830+p101*p841*p00+p01*p91*p100+p101*p01*p840+p91*p831*p00+p01*p121*p110+p111*p01*p850+p121*p861*p00+p01*p111*p120+p121*p01*p860+p111*p851*p00+p01*p141*p130+p131*p01*p870+p141*p881*p00+p01*p131*p140+p141*p01*p880+p131*p871*p00+p01*p161*p150+p151*p01*p890+p161*p901*p00+p01*p151*p160+p161*p01*p900+p151*p891*p00+p01*p181*p170+p171*p01*p910+p181*p921*p00+p01*p171*p180+p181*p01*p920+p171*p911*p00+p01*p201*p190+p191*p01*p930+p201*p941*p00+p01*p191*p200+p201*p01*p940+p191*p931*p00)+0.0242497*(p00*p101*p91+p90*p01*p831+p100*p841*p01+p00*p121*p111+p110*p01*p851+p120*p861*p01+p00*p141*p131+p130*p01*p871+p140*p881*p01+p00*p161*p151+p150*p01*p891+p160*p901*p01+p00*p181*p171+p170*p01*p911+p180*p921*p01+p00*p201*p191+p190*p01*p931+p200*p941*p01)-0.000944625*(p00*p100*p90+p90*p00*p830+p100*p840*p00+p00*p120*p110+p110*p00*p850+p120*p860*p00+p00*p140*p130+p130*p00*p870+p140*p880*p00+p00*p160*p150+p150*p00*p890+p160*p900*p00+p00*p180*p170+p170*p00*p910+p180*p920*p00+p00*p200*p190+p190*p00*p930+p200*p940*p00)+0.00403165*(p450*p01*p890+p490*p901*p00+p490*p01*p900+p450*p891*p00+p470*p01*p930+p490*p941*p00+p490*p01*p940+p470*p931*p00+p390*p01*p850+p500*p861*p00+p500*p01*p860+p390*p851*p00+p430*p01*p870+p500*p881*p00+p500*p01*p880+p430*p871*p00+p450*p01*p910+p470*p921*p00+p470*p01*p920+p450*p911*p00+p370*p01*p830+p480*p841*p00+p480*p01*p840+p370*p831*p00+p410*p01*p870+p480*p881*p00+p480*p01*p880+p410*p871*p00+p330*p01*p830+p460*p841*p00+p460*p01*p840+p330*p831*p00+p350*p01*p850+p460*p861*p00+p460*p01*p860+p350*p851*p00+p390*p01*p910+p430*p921*p00+p430*p01*p920+p390*p911*p00+p310*p01*p830+p440*p841*p00+p440*p01*p840+p310*p831*p00+p420*p01*p930+p440*p941*p00+p440*p01*p940+p420*p931*p00+p370*p01*p890+p410*p901*p00+p410*p01*p900+p370*p891*p00+p310*p01*p850+p420*p861*p00+p420*p01*p860+p310*p851*p00+p290*p01*p830+p400*p841*p00+p400*p01*p840+p290*p831*p00+p360*p01*p890+p400*p901*p00+p400*p01*p900+p360*p891*p00+p270*p01*p850+p380*p861*p00+p380*p01*p860+p270*p851*p00+p340*p01*p910+p380*p921*p00+p380*p01*p920+p340*p911*p00+p330*p01*p930+p350*p941*p00+p350*p01*p940+p330*p931*p00+p290*p01*p870+p360*p881*p00+p360*p01*p880+p290*p871*p00+p270*p01*p870+p340*p881*p00+p340*p01*p880+p270*p871*p00+p280*p01*p890+p320*p901*p00+p320*p01*p900+p280*p891*p00+p300*p01*p910+p320*p921*p00+p320*p01*p920+p300*p911*p00+p280*p01*p930+p300*p941*p00+p300*p01*p940+p280*p931*p00)+0.00119319*(p01*p90*p630+p101*p00*p860+p641*p850*p00+p01*p90*p670+p101*p00*p880+p681*p870*p00+p01*p90*p680+p101*p00*p900+p671*p890*p00+p01*p90*p640+p101*p00*p940+p631*p930*p00+p01*p110*p580+p121*p00*p840+p571*p830*p00+p01*p110*p660+p121*p00*p880+p651*p870*p00+p01*p110*p650+p121*p00*p920+p661*p910*p00+p01*p110*p570+p121*p00*p930+p581*p940*p00+p01*p130*p560+p141*p00*p840+p551*p830*p00+p01*p130*p620+p141*p00*p860+p611*p850*p00+p01*p130*p610+p141*p00*p910+p621*p920*p00+p01*p130*p550+p141*p00*p890+p561*p900*p00+p01*p150*p540+p161*p00*p840+p531*p830*p00+p01*p150*p740+p161*p00*p920+p731*p910*p00+p01*p150*p730+p161*p00*p940+p741*p930*p00+p01*p150*p530+p161*p00*p870+p541*p880*p00+p01*p170*p600+p181*p00*p860+p591*p850*p00+p01*p170*p720+p181*p00*p900+p711*p890*p00+p01*p170*p710+p181*p00*p930+p721*p940*p00+p01*p170*p590+p181*p00*p870+p601*p880*p00+p01*p190*p520+p201*p00*p840+p511*p830*p00+p01*p190*p700+p201*p00*p900+p691*p890*p00+p01*p190*p690+p201*p00*p910+p701*p920*p00+p01*p190*p510+p201*p00*p850+p521*p860*p00+p01*p200*p520+p191*p00*p860+p511*p850*p00+p01*p200*p700+p191*p00*p920+p691*p910*p00+p01*p200*p690+p191*p00*p890+p701*p900*p00+p01*p200*p510+p191*p00*p830+p521*p840*p00+p01*p180*p600+p171*p00*p880+p591*p870*p00+p01*p180*p720+p171*p00*p940+p711*p930*p00+p01*p180*p710+p171*p00*p890+p721*p900*p00+p01*p180*p590+p171*p00*p850+p601*p860*p00+p01*p160*p540+p151*p00*p880+p531*p870*p00+p01*p160*p740+p151*p00*p930+p731*p940*p00+p01*p160*p730+p151*p00*p910+p741*p920*p00+p01*p160*p530+p151*p00*p830+p541*p840*p00+p01*p140*p560+p131*p00*p900+p551*p890*p00+p01*p140*p620+p131*p00*p920+p611*p910*p00+p01*p140*p610+p131*p00*p850+p621*p860*p00+p01*p140*p550+p131*p00*p830+p561*p840*p00+p01*p120*p580+p111*p00*p940+p571*p930*p00+p01*p120*p660+p111*p00*p910+p651*p920*p00+p01*p120*p650+p111*p00*p870+p661*p880*p00+p01*p120*p570+p111*p00*p830+p581*p840*p00+p01*p100*p630+p91*p00*p930+p641*p940*p00+p01*p100*p670+p91*p00*p890+p681*p900*p00+p01*p100*p680+p91*p00*p870+p671*p880*p00+p01*p100*p640+p91*p00*p850+p631*p860*p00)+0.000984511*(p00*p91*p630+p100*p01*p860+p640*p851*p00+p00*p91*p670+p100*p01*p880+p680*p871*p00+p00*p91*p680+p100*p01*p900+p670*p891*p00+p00*p91*p640+p100*p01*p940+p630*p931*p00+p00*p111*p580+p120*p01*p840+p570*p831*p00+p00*p111*p660+p120*p01*p880+p650*p871*p00+p00*p111*p650+p120*p01*p920+p660*p911*p00+p00*p111*p570+p120*p01*p930+p580*p941*p00+p00*p131*p560+p140*p01*p840+p550*p831*p00+p00*p131*p620+p140*p01*p860+p610*p851*p00+p00*p131*p610+p140*p01*p910+p620*p921*p00+p00*p131*p550+p140*p01*p890+p560*p901*p00+p00*p151*p540+p160*p01*p840+p530*p831*p00+p00*p151*p740+p160*p01*p920+p730*p911*p00+p00*p151*p730+p160*p01*p940+p740*p931*p00+p00*p151*p530+p160*p01*p870+p540*p881*p00+p00*p171*p600+p180*p01*p860+p590*p851*p00+p00*p171*p720+p180*p01*p900+p710*p891*p00+p00*p171*p710+p180*p01*p930+p720*p941*p00+p00*p171*p590+p180*p01*p870+p600*p881*p00+p00*p191*p520+p200*p01*p840+p510*p831*p00+p00*p191*p700+p200*p01*p900+p690*p891*p00+p00*p191*p690+p200*p01*p910+p700*p921*p00+p00*p191*p510+p200*p01*p850+p520*p861*p00+p00*p201*p520+p190*p01*p860+p510*p851*p00+p00*p201*p700+p190*p01*p920+p690*p911*p00+p00*p201*p690+p190*p01*p890+p700*p901*p00+p00*p201*p510+p190*p01*p830+p520*p841*p00+p00*p181*p600+p170*p01*p880+p590*p871*p00+p00*p181*p720+p170*p01*p940+p710*p931*p00+p00*p181*p710+p170*p01*p890+p720*p901*p00+p00*p181*p590+p170*p01*p850+p600*p861*p00+p00*p161*p540+p150*p01*p880+p530*p871*p00+p00*p161*p740+p150*p01*p930+p730*p941*p00+p00*p161*p730+p150*p01*p910+p740*p921*p00+p00*p161*p530+p150*p01*p830+p540*p841*p00+p00*p141*p560+p130*p01*p900+p550*p891*p00+p00*p141*p620+p130*p01*p920+p610*p911*p00+p00*p141*p610+p130*p01*p850+p620*p861*p00+p00*p141*p550+p130*p01*p830+p560*p841*p00+p00*p121*p580+p110*p01*p940+p570*p931*p00+p00*p121*p660+p110*p01*p910+p650*p921*p00+p00*p121*p650+p110*p01*p870+p660*p881*p00+p00*p121*p570+p110*p01*p830+p580*p841*p00+p00*p101*p630+p90*p01*p930+p640*p941*p00+p00*p101*p670+p90*p01*p890+p680*p901*p00+p00*p101*p680+p90*p01*p870+p670*p881*p00+p00*p101*p640+p90*p01*p850+p630*p861*p00)-0.00162238*(p00*p90*p631+p100*p00*p861+p640*p850*p01+p00*p90*p671+p100*p00*p881+p680*p870*p01+p00*p90*p681+p100*p00*p901+p670*p890*p01+p00*p90*p641+p100*p00*p941+p630*p930*p01+p00*p110*p581+p120*p00*p841+p570*p830*p01+p00*p110*p661+p120*p00*p881+p650*p870*p01+p00*p110*p651+p120*p00*p921+p660*p910*p01+p00*p110*p571+p120*p00*p931+p580*p940*p01+p00*p130*p561+p140*p00*p841+p550*p830*p01+p00*p130*p621+p140*p00*p861+p610*p850*p01+p00*p130*p611+p140*p00*p911+p620*p920*p01+p00*p130*p551+p140*p00*p891+p560*p900*p01+p00*p150*p541+p160*p00*p841+p530*p830*p01+p00*p150*p741+p160*p00*p921+p730*p910*p01+p00*p150*p731+p160*p00*p941+p740*p930*p01+p00*p150*p531+p160*p00*p871+p540*p880*p01+p00*p170*p601+p180*p00*p861+p590*p850*p01+p00*p170*p721+p180*p00*p901+p710*p890*p01+p00*p170*p711+p180*p00*p931+p720*p940*p01+p00*p170*p591+p180*p00*p871+p600*p880*p01+p00*p190*p521+p200*p00*p841+p510*p830*p01+p00*p190*p701+p200*p00*p901+p690*p890*p01+p00*p190*p691+p200*p00*p911+p700*p920*p01+p00*p190*p511+p200*p00*p851+p520*p860*p01+p00*p200*p521+p190*p00*p861+p510*p850*p01+p00*p200*p701+p190*p00*p921+p690*p910*p01+p00*p200*p691+p190*p00*p891+p700*p900*p01+p00*p200*p511+p190*p00*p831+p520*p840*p01+p00*p180*p601+p170*p00*p881+p590*p870*p01+p00*p180*p721+p170*p00*p941+p710*p930*p01+p00*p180*p711+p170*p00*p891+p720*p900*p01+p00*p180*p591+p170*p00*p851+p600*p860*p01+p00*p160*p541+p150*p00*p881+p530*p870*p01+p00*p160*p741+p150*p00*p931+p730*p940*p01+p00*p160*p731+p150*p00*p911+p740*p920*p01+p00*p160*p531+p150*p00*p831+p540*p840*p01+p00*p140*p561+p130*p00*p901+p550*p890*p01+p00*p140*p621+p130*p00*p921+p610*p910*p01+p00*p140*p611+p130*p00*p851+p620*p860*p01+p00*p140*p551+p130*p00*p831+p560*p840*p01+p00*p120*p581+p110*p00*p941+p570*p930*p01+p00*p120*p661+p110*p00*p911+p650*p920*p01+p00*p120*p651+p110*p00*p871+p660*p880*p01+p00*p120*p571+p110*p00*p831+p580*p840*p01+p00*p100*p631+p90*p00*p931+p640*p940*p01+p00*p100*p671+p90*p00*p891+p680*p900*p01+p00*p100*p681+p90*p00*p871+p670*p880*p01+p00*p100*p641+p90*p00*p851+p630*p860*p01)-0.00346823*(p01*p91*p631+p101*p01*p861+p641*p851*p01+p01*p91*p671+p101*p01*p881+p681*p871*p01+p01*p91*p681+p101*p01*p901+p671*p891*p01+p01*p91*p641+p101*p01*p941+p631*p931*p01+p01*p111*p581+p121*p01*p841+p571*p831*p01+p01*p111*p661+p121*p01*p881+p651*p871*p01+p01*p111*p651+p121*p01*p921+p661*p911*p01+p01*p111*p571+p121*p01*p931+p581*p941*p01+p01*p131*p561+p141*p01*p841+p551*p831*p01+p01*p131*p621+p141*p01*p861+p611*p851*p01+p01*p131*p611+p141*p01*p911+p621*p921*p01+p01*p131*p551+p141*p01*p891+p561*p901*p01+p01*p151*p541+p161*p01*p841+p531*p831*p01+p01*p151*p741+p161*p01*p921+p731*p911*p01+p01*p151*p731+p161*p01*p941+p741*p931*p01+p01*p151*p531+p161*p01*p871+p541*p881*p01+p01*p171*p601+p181*p01*p861+p591*p851*p01+p01*p171*p721+p181*p01*p901+p711*p891*p01+p01*p171*p711+p181*p01*p931+p721*p941*p01+p01*p171*p591+p181*p01*p871+p601*p881*p01+p01*p191*p521+p201*p01*p841+p511*p831*p01+p01*p191*p701+p201*p01*p901+p691*p891*p01+p01*p191*p691+p201*p01*p911+p701*p921*p01+p01*p191*p511+p201*p01*p851+p521*p861*p01+p01*p201*p521+p191*p01*p861+p511*p851*p01+p01*p201*p701+p191*p01*p921+p691*p911*p01+p01*p201*p691+p191*p01*p891+p701*p901*p01+p01*p201*p511+p191*p01*p831+p521*p841*p01+p01*p181*p601+p171*p01*p881+p591*p871*p01+p01*p181*p721+p171*p01*p941+p711*p931*p01+p01*p181*p711+p171*p01*p891+p721*p901*p01+p01*p181*p591+p171*p01*p851+p601*p861*p01+p01*p161*p541+p151*p01*p881+p531*p871*p01+p01*p161*p741+p151*p01*p931+p731*p941*p01+p01*p161*p731+p151*p01*p911+p741*p921*p01+p01*p161*p531+p151*p01*p831+p541*p841*p01+p01*p141*p561+p131*p01*p901+p551*p891*p01+p01*p141*p621+p131*p01*p921+p611*p911*p01+p01*p141*p611+p131*p01*p851+p621*p861*p01+p01*p141*p551+p131*p01*p831+p561*p841*p01+p01*p121*p581+p111*p01*p941+p571*p931*p01+p01*p121*p661+p111*p01*p911+p651*p921*p01+p01*p121*p651+p111*p01*p871+p661*p881*p01+p01*p121*p571+p111*p01*p831+p581*p841*p01+p01*p101*p631+p91*p01*p931+p641*p941*p01+p01*p101*p671+p91*p01*p891+p681*p901*p01+p01*p101*p681+p91*p01*p871+p671*p881*p01+p01*p101*p641+p91*p01*p851+p631*p861*p01)+0.000211357*(p00*p90*p630+p100*p00*p860+p640*p850*p00+p00*p90*p670+p100*p00*p880+p680*p870*p00+p00*p90*p680+p100*p00*p900+p670*p890*p00+p00*p90*p640+p100*p00*p940+p630*p930*p00+p00*p110*p580+p120*p00*p840+p570*p830*p00+p00*p110*p660+p120*p00*p880+p650*p870*p00+p00*p110*p650+p120*p00*p920+p660*p910*p00+p00*p110*p570+p120*p00*p930+p580*p940*p00+p00*p130*p560+p140*p00*p840+p550*p830*p00+p00*p130*p620+p140*p00*p860+p610*p850*p00+p00*p130*p610+p140*p00*p910+p620*p920*p00+p00*p130*p550+p140*p00*p890+p560*p900*p00+p00*p150*p540+p160*p00*p840+p530*p830*p00+p00*p150*p740+p160*p00*p920+p730*p910*p00+p00*p150*p730+p160*p00*p940+p740*p930*p00+p00*p150*p530+p160*p00*p870+p540*p880*p00+p00*p170*p600+p180*p00*p860+p590*p850*p00+p00*p170*p720+p180*p00*p900+p710*p890*p00+p00*p170*p710+p180*p00*p930+p720*p940*p00+p00*p170*p590+p180*p00*p870+p600*p880*p00+p00*p190*p520+p200*p00*p840+p510*p830*p00+p00*p190*p700+p200*p00*p900+p690*p890*p00+p00*p190*p690+p200*p00*p910+p700*p920*p00+p00*p190*p510+p200*p00*p850+p520*p860*p00+p00*p200*p520+p190*p00*p860+p510*p850*p00+p00*p200*p700+p190*p00*p920+p690*p910*p00+p00*p200*p690+p190*p00*p890+p700*p900*p00+p00*p200*p510+p190*p00*p830+p520*p840*p00+p00*p180*p600+p170*p00*p880+p590*p870*p00+p00*p180*p720+p170*p00*p940+p710*p930*p00+p00*p180*p710+p170*p00*p890+p720*p900*p00+p00*p180*p590+p170*p00*p850+p600*p860*p00+p00*p160*p540+p150*p00*p880+p530*p870*p00+p00*p160*p740+p150*p00*p930+p730*p940*p00+p00*p160*p730+p150*p00*p910+p740*p920*p00+p00*p160*p530+p150*p00*p830+p540*p840*p00+p00*p140*p560+p130*p00*p900+p550*p890*p00+p00*p140*p620+p130*p00*p920+p610*p910*p00+p00*p140*p610+p130*p00*p850+p620*p860*p00+p00*p140*p550+p130*p00*p830+p560*p840*p00+p00*p120*p580+p110*p00*p940+p570*p930*p00+p00*p120*p660+p110*p00*p910+p650*p920*p00+p00*p120*p650+p110*p00*p870+p660*p880*p00+p00*p120*p570+p110*p00*p830+p580*p840*p00+p00*p100*p630+p90*p00*p930+p640*p940*p00+p00*p100*p670+p90*p00*p890+p680*p900*p00+p00*p100*p680+p90*p00*p870+p670*p880*p00+p00*p100*p640+p90*p00*p850+p630*p860*p00)+0.0038304*(p01*p610*p940+p621*p00*p560+p931*p550*p00+p01*p550*p930+p561*p00*p620+p941*p610*p00+p01*p650*p900+p661*p00*p580+p891*p570*p00+p01*p570*p890+p581*p00*p660+p901*p650*p00+p01*p690*p880+p701*p00*p520+p871*p510*p00+p01*p510*p870+p521*p00*p700+p881*p690*p00+p01*p740*p860+p731*p00*p540+p851*p530*p00+p01*p530*p850+p541*p00*p730+p861*p740*p00+p01*p680*p920+p671*p00*p630+p911*p640*p00+p01*p640*p910+p631*p00*p670+p921*p680*p00+p01*p720*p840+p711*p00*p600+p831*p590*p00+p01*p590*p830+p601*p00*p710+p841*p720*p00+p01*p630*p920+p641*p00*p680+p911*p670*p00+p01*p670*p910+p681*p00*p640+p921*p630*p00+p01*p600*p840+p591*p00*p720+p831*p710*p00+p01*p710*p830+p721*p00*p590+p841*p600*p00+p01*p580*p900+p571*p00*p650+p891*p660*p00+p01*p660*p890+p651*p00*p570+p901*p580*p00+p01*p540*p860+p531*p00*p740+p851*p730*p00+p01*p730*p850+p741*p00*p530+p861*p540*p00+p01*p560*p940+p551*p00*p610+p931*p620*p00+p01*p620*p930+p611*p00*p550+p941*p560*p00+p01*p520*p880+p511*p00*p690+p871*p700*p00+p01*p700*p870+p691*p00*p510+p881*p520*p00)-0.00154554*(p00*p611*p940+p620*p01*p560+p930*p551*p00+p00*p651*p900+p660*p01*p580+p890*p571*p00+p00*p691*p880+p700*p01*p520+p870*p511*p00+p00*p741*p860+p730*p01*p540+p850*p531*p00+p00*p681*p920+p670*p01*p630+p910*p641*p00+p00*p721*p840+p710*p01*p600+p830*p591*p00+p00*p631*p920+p640*p01*p680+p910*p671*p00+p00*p601*p840+p590*p01*p720+p830*p711*p00+p00*p581*p900+p570*p01*p650+p890*p661*p00+p00*p541*p860+p530*p01*p740+p850*p731*p00+p00*p561*p940+p550*p01*p610+p930*p621*p00+p00*p521*p880+p510*p01*p690+p870*p701*p00)-0.0023209*(p01*p610*p941+p621*p00*p561+p931*p550*p01+p01*p650*p901+p661*p00*p581+p891*p570*p01+p01*p690*p881+p701*p00*p521+p871*p510*p01+p01*p740*p861+p731*p00*p541+p851*p530*p01+p01*p680*p921+p671*p00*p631+p911*p640*p01+p01*p720*p841+p711*p00*p601+p831*p590*p01+p01*p630*p921+p641*p00*p681+p911*p670*p01+p01*p600*p841+p591*p00*p721+p831*p710*p01+p01*p580*p901+p571*p00*p651+p891*p660*p01+p01*p540*p861+p531*p00*p741+p851*p730*p01+p01*p560*p941+p551*p00*p611+p931*p620*p01+p01*p520*p881+p511*p00*p691+p871*p700*p01)+0.00752972*(p01*p611*p941+p621*p01*p561+p931*p551*p01+p01*p651*p901+p661*p01*p581+p891*p571*p01+p01*p691*p881+p701*p01*p521+p871*p511*p01+p01*p741*p861+p731*p01*p541+p851*p531*p01+p01*p681*p921+p671*p01*p631+p911*p641*p01+p01*p721*p841+p711*p01*p601+p831*p591*p01+p01*p631*p921+p641*p01*p681+p911*p671*p01+p01*p601*p841+p591*p01*p721+p831*p711*p01+p01*p581*p901+p571*p01*p651+p891*p661*p01+p01*p541*p861+p531*p01*p741+p851*p731*p01+p01*p561*p941+p551*p01*p611+p931*p621*p01+p01*p521*p881+p511*p01*p691+p871*p701*p01)+8.24539e-05*(p00*p610*p940+p620*p00*p560+p930*p550*p00+p00*p650*p900+p660*p00*p580+p890*p570*p00+p00*p690*p880+p700*p00*p520+p870*p510*p00+p00*p740*p860+p730*p00*p540+p850*p530*p00+p00*p680*p920+p670*p00*p630+p910*p640*p00+p00*p720*p840+p710*p00*p600+p830*p590*p00+p00*p630*p920+p640*p00*p680+p910*p670*p00+p00*p600*p840+p590*p00*p720+p830*p710*p00+p00*p580*p900+p570*p00*p650+p890*p660*p00+p00*p540*p860+p530*p00*p740+p850*p730*p00+p00*p560*p940+p550*p00*p610+p930*p620*p00+p00*p520*p880+p510*p00*p690+p870*p700*p00)-0.000189267*(p01*p840*p860+p831*p00*p930+p851*p940*p00+p01*p940*p850+p931*p00*p830+p861*p840*p00+p01*p930*p830+p941*p00*p850+p841*p860*p00+p01*p840*p880+p831*p00*p890+p871*p900*p00+p01*p900*p870+p891*p00*p830+p881*p840*p00+p01*p890*p830+p901*p00*p870+p841*p880*p00+p01*p840*p900+p831*p00*p870+p891*p880*p00+p01*p880*p890+p871*p00*p830+p901*p840*p00+p01*p870*p830+p881*p00*p890+p841*p900*p00+p01*p840*p940+p831*p00*p850+p931*p860*p00+p01*p860*p930+p851*p00*p830+p941*p840*p00+p01*p850*p830+p861*p00*p930+p841*p940*p00+p01*p860*p880+p851*p00*p910+p871*p920*p00+p01*p920*p870+p911*p00*p850+p881*p860*p00+p01*p910*p850+p921*p00*p870+p861*p880*p00+p01*p860*p920+p851*p00*p870+p911*p880*p00+p01*p880*p910+p871*p00*p850+p921*p860*p00+p01*p870*p850+p881*p00*p910+p861*p920*p00+p01*p900*p920+p891*p00*p930+p911*p940*p00+p01*p940*p910+p931*p00*p890+p921*p900*p00+p01*p930*p890+p941*p00*p910+p901*p920*p00+p01*p900*p940+p891*p00*p910+p931*p920*p00+p01*p920*p930+p911*p00*p890+p941*p900*p00+p01*p910*p890+p921*p00*p930+p901*p940*p00)-0.000551881*(p01*p841*p860+p831*p01*p930+p851*p941*p00+p01*p861*p840+p851*p01*p940+p831*p931*p00+p01*p941*p850+p931*p01*p830+p861*p841*p00+p01*p841*p880+p831*p01*p890+p871*p901*p00+p01*p881*p840+p871*p01*p900+p831*p891*p00+p01*p901*p870+p891*p01*p830+p881*p841*p00+p01*p841*p900+p831*p01*p870+p891*p881*p00+p01*p881*p890+p871*p01*p830+p901*p841*p00+p01*p901*p840+p891*p01*p880+p831*p871*p00+p01*p841*p940+p831*p01*p850+p931*p861*p00+p01*p861*p930+p851*p01*p830+p941*p841*p00+p01*p941*p840+p931*p01*p860+p831*p851*p00+p01*p861*p880+p851*p01*p910+p871*p921*p00+p01*p921*p870+p911*p01*p850+p881*p861*p00+p01*p881*p860+p871*p01*p920+p851*p911*p00+p01*p861*p920+p851*p01*p870+p911*p881*p00+p01*p921*p860+p911*p01*p880+p851*p871*p00+p01*p881*p910+p871*p01*p850+p921*p861*p00+p01*p901*p920+p891*p01*p930+p911*p941*p00+p01*p921*p900+p911*p01*p940+p891*p931*p00+p01*p941*p910+p931*p01*p890+p921*p901*p00+p01*p901*p940+p891*p01*p910+p931*p921*p00+p01*p921*p930+p911*p01*p890+p941*p901*p00+p01*p941*p900+p931*p01*p920+p891*p911*p00)-0.00144141*(p01*p841*p861+p831*p01*p931+p851*p941*p01+p01*p841*p881+p831*p01*p891+p871*p901*p01+p01*p841*p901+p831*p01*p871+p891*p881*p01+p01*p841*p941+p831*p01*p851+p931*p861*p01+p01*p861*p881+p851*p01*p911+p871*p921*p01+p01*p861*p921+p851*p01*p871+p911*p881*p01+p01*p901*p921+p891*p01*p931+p911*p941*p01+p01*p901*p941+p891*p01*p911+p931*p921*p01)-0.00150974*(p01*p90*p190*p150+p101*p00*p120*p140+p201*p110*p00*p170+p161*p130*p180*p00+p01*p110*p200*p170+p121*p00*p100*p140+p191*p90*p00*p150+p181*p130*p160*p00+p01*p130*p160*p180+p141*p00*p100*p120+p151*p90*p00*p190+p171*p110*p200*p00+p01*p140*p100*p120+p131*p00*p160*p180+p91*p150*p00*p190+p111*p170*p200*p00+p01*p90*p130*p110+p101*p00*p160*p200+p141*p150*p00*p170+p121*p190*p180*p00+p01*p150*p140*p170+p161*p00*p100*p200+p131*p90*p00*p110+p181*p190*p120*p00+p01*p190*p120*p180+p201*p00*p100*p160+p111*p90*p00*p130+p171*p150*p140*p00+p01*p200*p100*p160+p191*p00*p120*p180+p91*p110*p00*p130+p151*p170*p140*p00)-0.00677057*(p01*p91*p191*p150+p101*p01*p121*p140+p201*p111*p01*p170+p161*p131*p181*p00+p01*p91*p151*p190+p101*p01*p141*p120+p161*p131*p01*p180+p201*p111*p171*p00+p01*p111*p171*p200+p121*p01*p141*p100+p181*p131*p01*p160+p191*p91*p151*p00+p01*p151*p191*p90+p161*p01*p181*p130+p201*p171*p01*p110+p101*p141*p121*p00+p01*p91*p131*p110+p101*p01*p161*p200+p141*p151*p01*p170+p121*p191*p181*p00+p01*p91*p111*p130+p101*p01*p201*p160+p121*p191*p01*p180+p141*p151*p171*p00+p01*p111*p131*p90+p121*p01*p181*p190+p141*p171*p01*p150+p101*p201*p161*p00+p01*p151*p171*p140+p161*p01*p201*p100+p181*p191*p01*p120+p131*p91*p111*p00)-0.0121651*(p01*p91*p191*p151+p101*p01*p121*p141+p201*p111*p01*p171+p161*p131*p181*p01+p01*p91*p131*p111+p101*p01*p161*p201+p141*p151*p01*p171+p121*p191*p181*p01);
     return energy;
  }


  if(b == 1){
     l=index(i,j,k,1);
     int p00=mcL[l];
     l=index(i,j,k,1);
     int p01=mcL[l]*mcL[l];
     l=index(i,j,k,0);
     int p10=mcL[l];
     l=index(i,j,k,0);
     int p11=mcL[l]*mcL[l];
     l=index(i,j,k+1,0);
     int p20=mcL[l];
     l=index(i,j,k+1,0);
     int p21=mcL[l]*mcL[l];
     l=index(i,j+1,k,0);
     int p30=mcL[l];
     l=index(i,j+1,k,0);
     int p31=mcL[l]*mcL[l];
     l=index(i+1,j,k,0);
     int p40=mcL[l];
     l=index(i+1,j,k,0);
     int p41=mcL[l]*mcL[l];
     l=index(i,j,k-1,2);
     int p50=mcL[l];
     l=index(i,j,k-1,2);
     int p51=mcL[l]*mcL[l];
     l=index(i,j-1,k,2);
     int p60=mcL[l];
     l=index(i,j-1,k,2);
     int p61=mcL[l]*mcL[l];
     l=index(i-1,j-1,k,2);
     int p70=mcL[l];
     l=index(i-1,j-1,k,2);
     int p71=mcL[l]*mcL[l];
     l=index(i-1,j,k-1,2);
     int p80=mcL[l];
     l=index(i-1,j,k-1,2);
     int p81=mcL[l]*mcL[l];
     l=index(i,j-1,k-1,2);
     int p90=mcL[l];
     l=index(i,j-1,k-1,2);
     int p91=mcL[l]*mcL[l];
     l=index(i-1,j,k,2);
     int p100=mcL[l];
     l=index(i-1,j,k,2);
     int p101=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,2);
     int p110=mcL[l];
     l=index(i+1,j-1,k-1,2);
     int p111=mcL[l]*mcL[l];
     l=index(i-1,j-1,k-1,2);
     int p120=mcL[l];
     l=index(i-1,j-1,k-1,2);
     int p121=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,2);
     int p130=mcL[l];
     l=index(i-1,j-1,k+1,2);
     int p131=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,2);
     int p140=mcL[l];
     l=index(i-1,j+1,k-1,2);
     int p141=mcL[l]*mcL[l];
     l=index(i-1,j,k,0);
     int p150=mcL[l];
     l=index(i-1,j,k,0);
     int p151=mcL[l]*mcL[l];
     l=index(i-1,j,k+2,0);
     int p160=mcL[l];
     l=index(i-1,j,k+2,0);
     int p161=mcL[l]*mcL[l];
     l=index(i-1,j+2,k,0);
     int p170=mcL[l];
     l=index(i-1,j+2,k,0);
     int p171=mcL[l]*mcL[l];
     l=index(i,j-1,k,0);
     int p180=mcL[l];
     l=index(i,j-1,k,0);
     int p181=mcL[l]*mcL[l];
     l=index(i,j-1,k+2,0);
     int p190=mcL[l];
     l=index(i,j-1,k+2,0);
     int p191=mcL[l]*mcL[l];
     l=index(i,j,k-1,0);
     int p200=mcL[l];
     l=index(i,j,k-1,0);
     int p201=mcL[l]*mcL[l];
     l=index(i,j,k+2,0);
     int p210=mcL[l];
     l=index(i,j,k+2,0);
     int p211=mcL[l]*mcL[l];
     l=index(i,j+2,k-1,0);
     int p220=mcL[l];
     l=index(i,j+2,k-1,0);
     int p221=mcL[l]*mcL[l];
     l=index(i,j+2,k,0);
     int p230=mcL[l];
     l=index(i,j+2,k,0);
     int p231=mcL[l]*mcL[l];
     l=index(i+2,j-1,k,0);
     int p240=mcL[l];
     l=index(i+2,j-1,k,0);
     int p241=mcL[l]*mcL[l];
     l=index(i+2,j,k-1,0);
     int p250=mcL[l];
     l=index(i+2,j,k-1,0);
     int p251=mcL[l]*mcL[l];
     l=index(i+2,j,k,0);
     int p260=mcL[l];
     l=index(i+2,j,k,0);
     int p261=mcL[l]*mcL[l];
     l=index(i-2,j+1,k+1,0);
     int p270=mcL[l];
     l=index(i-2,j+1,k+1,0);
     int p271=mcL[l]*mcL[l];
     l=index(i+1,j-2,k+1,0);
     int p280=mcL[l];
     l=index(i+1,j-2,k+1,0);
     int p281=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-2,0);
     int p290=mcL[l];
     l=index(i+1,j+1,k-2,0);
     int p291=mcL[l]*mcL[l];
     l=index(i+1,j+1,k+1,0);
     int p300=mcL[l];
     l=index(i+1,j+1,k+1,0);
     int p301=mcL[l]*mcL[l];
     l=index(i+2,j,k,1);
     int p310=mcL[l];
     l=index(i+2,j,k,1);
     int p311=mcL[l]*mcL[l];
     l=index(i-2,j,k,1);
     int p320=mcL[l];
     l=index(i-2,j,k,1);
     int p321=mcL[l]*mcL[l];
     l=index(i+2,j,k-2,1);
     int p330=mcL[l];
     l=index(i+2,j,k-2,1);
     int p331=mcL[l]*mcL[l];
     l=index(i-2,j,k+2,1);
     int p340=mcL[l];
     l=index(i-2,j,k+2,1);
     int p341=mcL[l]*mcL[l];
     l=index(i+2,j-2,k,1);
     int p350=mcL[l];
     l=index(i+2,j-2,k,1);
     int p351=mcL[l]*mcL[l];
     l=index(i-2,j+2,k,1);
     int p360=mcL[l];
     l=index(i-2,j+2,k,1);
     int p361=mcL[l]*mcL[l];
     l=index(i,j+2,k,1);
     int p370=mcL[l];
     l=index(i,j+2,k,1);
     int p371=mcL[l]*mcL[l];
     l=index(i,j-2,k,1);
     int p380=mcL[l];
     l=index(i,j-2,k,1);
     int p381=mcL[l]*mcL[l];
     l=index(i,j+2,k-2,1);
     int p390=mcL[l];
     l=index(i,j+2,k-2,1);
     int p391=mcL[l]*mcL[l];
     l=index(i,j-2,k+2,1);
     int p400=mcL[l];
     l=index(i,j-2,k+2,1);
     int p401=mcL[l]*mcL[l];
     l=index(i,j,k+2,1);
     int p410=mcL[l];
     l=index(i,j,k+2,1);
     int p411=mcL[l]*mcL[l];
     l=index(i,j,k-2,1);
     int p420=mcL[l];
     l=index(i,j,k-2,1);
     int p421=mcL[l]*mcL[l];
     l=index(i,j+1,k+1,0);
     int p430=mcL[l];
     l=index(i,j+1,k+1,0);
     int p431=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p440=mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p441=mcL[l]*mcL[l];
     l=index(i,j+1,k-1,0);
     int p450=mcL[l];
     l=index(i,j+1,k-1,0);
     int p451=mcL[l]*mcL[l];
     l=index(i-1,j+1,k,0);
     int p460=mcL[l];
     l=index(i-1,j+1,k,0);
     int p461=mcL[l]*mcL[l];
     l=index(i,j-1,k+1,0);
     int p470=mcL[l];
     l=index(i,j-1,k+1,0);
     int p471=mcL[l]*mcL[l];
     l=index(i-1,j,k+1,0);
     int p480=mcL[l];
     l=index(i-1,j,k+1,0);
     int p481=mcL[l]*mcL[l];
     l=index(i+1,j,k+1,0);
     int p490=mcL[l];
     l=index(i+1,j,k+1,0);
     int p491=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p500=mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p501=mcL[l]*mcL[l];
     l=index(i+1,j,k-1,0);
     int p510=mcL[l];
     l=index(i+1,j,k-1,0);
     int p511=mcL[l]*mcL[l];
     l=index(i+1,j-1,k,0);
     int p520=mcL[l];
     l=index(i+1,j-1,k,0);
     int p521=mcL[l]*mcL[l];
     l=index(i+1,j+1,k,0);
     int p530=mcL[l];
     l=index(i+1,j+1,k,0);
     int p531=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p540=mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p541=mcL[l]*mcL[l];
     l=index(i+1,j+2,k-1,0);
     int p550=mcL[l];
     l=index(i+1,j+2,k-1,0);
     int p551=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+2,0);
     int p560=mcL[l];
     l=index(i+1,j-1,k+2,0);
     int p561=mcL[l]*mcL[l];
     l=index(i+1,j-1,k-1,0);
     int p570=mcL[l];
     l=index(i+1,j-1,k-1,0);
     int p571=mcL[l]*mcL[l];
     l=index(i+2,j+1,k-1,0);
     int p580=mcL[l];
     l=index(i+2,j+1,k-1,0);
     int p581=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+2,0);
     int p590=mcL[l];
     l=index(i-1,j+1,k+2,0);
     int p591=mcL[l]*mcL[l];
     l=index(i-1,j+1,k-1,0);
     int p600=mcL[l];
     l=index(i-1,j+1,k-1,0);
     int p601=mcL[l]*mcL[l];
     l=index(i+2,j-1,k+1,0);
     int p610=mcL[l];
     l=index(i+2,j-1,k+1,0);
     int p611=mcL[l]*mcL[l];
     l=index(i-1,j+2,k+1,0);
     int p620=mcL[l];
     l=index(i-1,j+2,k+1,0);
     int p621=mcL[l]*mcL[l];
     l=index(i+2,j-1,k-1,0);
     int p630=mcL[l];
     l=index(i+2,j-1,k-1,0);
     int p631=mcL[l]*mcL[l];
     l=index(i-1,j+2,k-1,0);
     int p640=mcL[l];
     l=index(i-1,j+2,k-1,0);
     int p641=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+1,0);
     int p650=mcL[l];
     l=index(i-1,j-1,k+1,0);
     int p651=mcL[l]*mcL[l];
     l=index(i-1,j-1,k+2,0);
     int p660=mcL[l];
     l=index(i-1,j-1,k+2,0);
     int p661=mcL[l]*mcL[l];

     energy = energy-6.9586*(p01)+0.658756*(p00)-0.0173448*(p01*p10+p01*p20+p01*p30+p01*p40)-0.159011*(p00*p11+p00*p21+p00*p31+p00*p41)+0.679524*(p01*p11+p01*p21+p01*p31+p01*p41)+0.0841386*(p00*p10+p00*p20+p00*p30+p00*p40)-0.092346*(p01*p50+p51*p00+p01*p60+p61*p00+p71*p00+p01*p70+p81*p00+p01*p80+p91*p00+p01*p90+p01*p100+p101*p00)+0.275366*(p01*p51+p01*p61+p71*p01+p81*p01+p91*p01+p01*p101)+0.0551302*(p01*p111+p121*p01+p01*p131+p01*p141)-0.0138053*(p00*p110+p120*p00+p00*p130+p00*p140)-0.00161467*(p01*p151+p01*p161+p01*p171+p01*p181+p01*p191+p01*p201+p01*p211+p01*p221+p01*p231+p01*p241+p01*p251+p01*p261)-0.0383274*(p01*p270+p01*p280+p01*p290+p01*p300)+0.00311609*(p01*p271+p01*p281+p01*p291+p01*p301)+0.0421336*(p00*p270+p00*p280+p00*p290+p00*p300)+0.00812835*(p01*p310+p321*p00+p01*p320+p311*p00+p01*p330+p341*p00+p01*p340+p331*p00+p01*p350+p361*p00+p01*p360+p351*p00+p01*p370+p381*p00+p01*p380+p371*p00+p01*p390+p401*p00+p01*p400+p391*p00+p01*p410+p421*p00+p01*p420+p411*p00)+0.00956795*(p00*p310+p320*p00+p00*p330+p340*p00+p00*p350+p360*p00+p00*p370+p380*p00+p00*p390+p400*p00+p00*p410+p420*p00)-0.00450773*(p430*p440*p01+p450*p460*p01+p470*p480*p01+p490*p500*p01+p510*p520*p01+p530*p540*p01)-0.00212345*(p430*p440*p00+p450*p460*p00+p470*p480*p00+p490*p500*p00+p510*p520*p00+p530*p540*p00)-0.00158958*(p00*p40*p471+p00*p40*p451+p00*p40*p431+p00*p30*p481+p00*p30*p511+p00*p30*p491+p00*p20*p461+p00*p20*p521+p00*p10*p441+p00*p10*p501+p00*p20*p531+p00*p10*p541)-3.18114e-05*(p231*p170*p01+p171*p230*p01+p211*p160*p01+p161*p210*p01+p221*p170*p01+p171*p220*p01+p201*p150*p01+p151*p200*p01+p191*p160*p01+p161*p190*p01+p181*p150*p01+p151*p180*p01+p261*p240*p01+p241*p260*p01+p211*p190*p01+p191*p210*p01+p251*p240*p01+p241*p250*p01+p201*p180*p01+p181*p200*p01+p261*p250*p01+p251*p260*p01+p231*p220*p01+p221*p230*p01)+0.00780764*(p00*p531*p210+p00*p491*p230+p00*p541*p200+p00*p511*p220+p00*p501*p180+p00*p521*p190+p00*p431*p260+p00*p451*p250+p00*p471*p240+p00*p441*p150+p00*p461*p160+p00*p481*p170)-0.000457364*(p01*p40*p180+p01*p40*p200+p01*p40*p190+p01*p40*p210+p01*p40*p220+p01*p40*p230+p01*p30*p150+p01*p30*p200+p01*p30*p160+p01*p30*p210+p01*p20*p150+p01*p20*p180+p01*p10*p160+p01*p10*p190+p01*p20*p170+p01*p20*p230+p01*p10*p170+p01*p10*p220+p01*p30*p250+p01*p30*p260+p01*p20*p240+p01*p20*p260+p01*p10*p240+p01*p10*p250)+0.00433732*(p141*p550*p00+p131*p560*p00+p01*p530*p130+p01*p490*p140+p121*p570*p00+p01*p540*p120+p01*p510*p140+p01*p500*p120+p01*p520*p130+p111*p580*p00+p01*p430*p110+p131*p590*p00+p01*p450*p110+p121*p600*p00+p111*p610*p00+p141*p620*p00+p111*p630*p00+p141*p640*p00+p01*p470*p110+p121*p650*p00+p131*p660*p00+p01*p440*p120+p01*p460*p130+p01*p480*p140)+0.00518956*(p140*p551*p00+p130*p561*p00+p00*p531*p130+p00*p491*p140+p120*p571*p00+p00*p541*p120+p00*p511*p140+p00*p501*p120+p00*p521*p130+p110*p581*p00+p00*p431*p110+p130*p591*p00+p00*p451*p110+p120*p601*p00+p110*p611*p00+p140*p621*p00+p110*p631*p00+p140*p641*p00+p00*p471*p110+p120*p651*p00+p130*p661*p00+p00*p441*p120+p00*p461*p130+p00*p481*p140)+0.0118966*(p140*p550*p01+p130*p560*p01+p00*p530*p131+p00*p490*p141+p120*p570*p01+p00*p540*p121+p00*p510*p141+p00*p500*p121+p00*p520*p131+p110*p580*p01+p00*p430*p111+p130*p590*p01+p00*p450*p111+p120*p600*p01+p110*p610*p01+p140*p620*p01+p110*p630*p01+p140*p640*p01+p00*p470*p111+p120*p650*p01+p130*p660*p01+p00*p440*p121+p00*p460*p131+p00*p480*p141)+0.00789307*(p221*p550*p00+p191*p560*p00+p231*p550*p00+p181*p570*p00+p211*p560*p00+p201*p570*p00+p251*p580*p00+p161*p590*p00+p261*p580*p00+p151*p600*p00+p241*p610*p00+p171*p620*p00+p241*p630*p00+p171*p640*p00+p261*p610*p00+p151*p650*p00+p251*p630*p00+p161*p660*p00+p211*p590*p00+p201*p600*p00+p231*p620*p00+p181*p650*p00+p221*p640*p00+p191*p660*p00)+0.0011537*(p220*p551*p00+p190*p561*p00+p230*p551*p00+p180*p571*p00+p210*p561*p00+p200*p571*p00+p250*p581*p00+p160*p591*p00+p260*p581*p00+p150*p601*p00+p240*p611*p00+p170*p621*p00+p240*p631*p00+p170*p641*p00+p260*p611*p00+p150*p651*p00+p250*p631*p00+p160*p661*p00+p210*p591*p00+p200*p601*p00+p230*p621*p00+p180*p651*p00+p220*p641*p00+p190*p661*p00)+0.00403165*(p00*p211*p260+p00*p261*p210+p00*p231*p260+p00*p261*p230+p00*p201*p250+p00*p251*p200+p00*p221*p250+p00*p251*p220+p00*p181*p240+p00*p241*p180+p00*p191*p240+p00*p241*p190+p00*p211*p230+p00*p231*p210+p00*p201*p220+p00*p221*p200+p00*p181*p190+p00*p191*p180+p00*p151*p170+p00*p171*p150+p00*p161*p170+p00*p171*p160+p00*p151*p160+p00*p161*p150);
     return energy;
  }


  if(b == 2){
     l=index(i,j,k,2);
     int p00=mcL[l];
     l=index(i,j,k,2);
     int p01=mcL[l]*mcL[l];
     l=index(i,j+1,k+1,0);
     int p10=mcL[l];
     l=index(i,j+1,k+1,0);
     int p11=mcL[l]*mcL[l];
     l=index(i+1,j,k+1,0);
     int p20=mcL[l];
     l=index(i+1,j,k+1,0);
     int p21=mcL[l]*mcL[l];
     l=index(i+1,j+1,k,0);
     int p30=mcL[l];
     l=index(i+1,j+1,k,0);
     int p31=mcL[l]*mcL[l];
     l=index(i+1,j+1,k+1,0);
     int p40=mcL[l];
     l=index(i+1,j+1,k+1,0);
     int p41=mcL[l]*mcL[l];
     l=index(i,j,k+1,1);
     int p50=mcL[l];
     l=index(i,j,k+1,1);
     int p51=mcL[l]*mcL[l];
     l=index(i,j+1,k,1);
     int p60=mcL[l];
     l=index(i,j+1,k,1);
     int p61=mcL[l]*mcL[l];
     l=index(i+1,j+1,k,1);
     int p70=mcL[l];
     l=index(i+1,j+1,k,1);
     int p71=mcL[l]*mcL[l];
     l=index(i+1,j,k+1,1);
     int p80=mcL[l];
     l=index(i+1,j,k+1,1);
     int p81=mcL[l]*mcL[l];
     l=index(i,j+1,k+1,1);
     int p90=mcL[l];
     l=index(i,j+1,k+1,1);
     int p91=mcL[l]*mcL[l];
     l=index(i+1,j,k,1);
     int p100=mcL[l];
     l=index(i+1,j,k,1);
     int p101=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,1);
     int p110=mcL[l];
     l=index(i-1,j+1,k+1,1);
     int p111=mcL[l]*mcL[l];
     l=index(i+1,j+1,k+1,1);
     int p120=mcL[l];
     l=index(i+1,j+1,k+1,1);
     int p121=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,1);
     int p130=mcL[l];
     l=index(i+1,j+1,k-1,1);
     int p131=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,1);
     int p140=mcL[l];
     l=index(i+1,j-1,k+1,1);
     int p141=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p150=mcL[l];
     l=index(i-1,j+1,k+1,0);
     int p151=mcL[l]*mcL[l];
     l=index(i-1,j+1,k+2,0);
     int p160=mcL[l];
     l=index(i-1,j+1,k+2,0);
     int p161=mcL[l]*mcL[l];
     l=index(i-1,j+2,k+1,0);
     int p170=mcL[l];
     l=index(i-1,j+2,k+1,0);
     int p171=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p180=mcL[l];
     l=index(i+1,j-1,k+1,0);
     int p181=mcL[l]*mcL[l];
     l=index(i+1,j-1,k+2,0);
     int p190=mcL[l];
     l=index(i+1,j-1,k+2,0);
     int p191=mcL[l]*mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p200=mcL[l];
     l=index(i+1,j+1,k-1,0);
     int p201=mcL[l]*mcL[l];
     l=index(i+1,j+1,k+2,0);
     int p210=mcL[l];
     l=index(i+1,j+1,k+2,0);
     int p211=mcL[l]*mcL[l];
     l=index(i+1,j+2,k-1,0);
     int p220=mcL[l];
     l=index(i+1,j+2,k-1,0);
     int p221=mcL[l]*mcL[l];
     l=index(i+1,j+2,k+1,0);
     int p230=mcL[l];
     l=index(i+1,j+2,k+1,0);
     int p231=mcL[l]*mcL[l];
     l=index(i+2,j-1,k+1,0);
     int p240=mcL[l];
     l=index(i+2,j-1,k+1,0);
     int p241=mcL[l]*mcL[l];
     l=index(i+2,j+1,k-1,0);
     int p250=mcL[l];
     l=index(i+2,j+1,k-1,0);
     int p251=mcL[l]*mcL[l];
     l=index(i+2,j+1,k+1,0);
     int p260=mcL[l];
     l=index(i+2,j+1,k+1,0);
     int p261=mcL[l]*mcL[l];
     l=index(i,j,k,0);
     int p270=mcL[l];
     l=index(i,j,k,0);
     int p271=mcL[l]*mcL[l];
     l=index(i,j,k+3,0);
     int p280=mcL[l];
     l=index(i,j,k+3,0);
     int p281=mcL[l]*mcL[l];
     l=index(i,j+3,k,0);
     int p290=mcL[l];
     l=index(i,j+3,k,0);
     int p291=mcL[l]*mcL[l];
     l=index(i+3,j,k,0);
     int p300=mcL[l];
     l=index(i+3,j,k,0);
     int p301=mcL[l]*mcL[l];
     l=index(i+2,j,k,2);
     int p310=mcL[l];
     l=index(i+2,j,k,2);
     int p311=mcL[l]*mcL[l];
     l=index(i-2,j,k,2);
     int p320=mcL[l];
     l=index(i-2,j,k,2);
     int p321=mcL[l]*mcL[l];
     l=index(i+2,j,k-2,2);
     int p330=mcL[l];
     l=index(i+2,j,k-2,2);
     int p331=mcL[l]*mcL[l];
     l=index(i-2,j,k+2,2);
     int p340=mcL[l];
     l=index(i-2,j,k+2,2);
     int p341=mcL[l]*mcL[l];
     l=index(i+2,j-2,k,2);
     int p350=mcL[l];
     l=index(i+2,j-2,k,2);
     int p351=mcL[l]*mcL[l];
     l=index(i-2,j+2,k,2);
     int p360=mcL[l];
     l=index(i-2,j+2,k,2);
     int p361=mcL[l]*mcL[l];
     l=index(i,j+2,k,2);
     int p370=mcL[l];
     l=index(i,j+2,k,2);
     int p371=mcL[l]*mcL[l];
     l=index(i,j-2,k,2);
     int p380=mcL[l];
     l=index(i,j-2,k,2);
     int p381=mcL[l]*mcL[l];
     l=index(i,j+2,k-2,2);
     int p390=mcL[l];
     l=index(i,j+2,k-2,2);
     int p391=mcL[l]*mcL[l];
     l=index(i,j-2,k+2,2);
     int p400=mcL[l];
     l=index(i,j-2,k+2,2);
     int p401=mcL[l]*mcL[l];
     l=index(i,j,k+2,2);
     int p410=mcL[l];
     l=index(i,j,k+2,2);
     int p411=mcL[l]*mcL[l];
     l=index(i,j,k-2,2);
     int p420=mcL[l];
     l=index(i,j,k-2,2);
     int p421=mcL[l]*mcL[l];
     l=index(i+2,j,k,0);
     int p430=mcL[l];
     l=index(i+2,j,k,0);
     int p431=mcL[l]*mcL[l];
     l=index(i+1,j,k,0);
     int p440=mcL[l];
     l=index(i+1,j,k,0);
     int p441=mcL[l]*mcL[l];
     l=index(i+2,j,k+1,0);
     int p450=mcL[l];
     l=index(i+2,j,k+1,0);
     int p451=mcL[l]*mcL[l];
     l=index(i+1,j,k+2,0);
     int p460=mcL[l];
     l=index(i+1,j,k+2,0);
     int p461=mcL[l]*mcL[l];
     l=index(i+2,j+1,k,0);
     int p470=mcL[l];
     l=index(i+2,j+1,k,0);
     int p471=mcL[l]*mcL[l];
     l=index(i+1,j+2,k,0);
     int p480=mcL[l];
     l=index(i+1,j+2,k,0);
     int p481=mcL[l]*mcL[l];
     l=index(i,j+2,k,0);
     int p490=mcL[l];
     l=index(i,j+2,k,0);
     int p491=mcL[l]*mcL[l];
     l=index(i,j+1,k,0);
     int p500=mcL[l];
     l=index(i,j+1,k,0);
     int p501=mcL[l]*mcL[l];
     l=index(i,j+2,k+1,0);
     int p510=mcL[l];
     l=index(i,j+2,k+1,0);
     int p511=mcL[l]*mcL[l];
     l=index(i,j+1,k+2,0);
     int p520=mcL[l];
     l=index(i,j+1,k+2,0);
     int p521=mcL[l]*mcL[l];
     l=index(i,j,k+2,0);
     int p530=mcL[l];
     l=index(i,j,k+2,0);
     int p531=mcL[l]*mcL[l];
     l=index(i,j,k+1,0);
     int p540=mcL[l];
     l=index(i,j,k+1,0);
     int p541=mcL[l]*mcL[l];
     l=index(i+2,j+2,k-1,0);
     int p550=mcL[l];
     l=index(i+2,j+2,k-1,0);
     int p551=mcL[l]*mcL[l];
     l=index(i+2,j-1,k+2,0);
     int p560=mcL[l];
     l=index(i+2,j-1,k+2,0);
     int p561=mcL[l]*mcL[l];
     l=index(i+2,j+2,k,0);
     int p570=mcL[l];
     l=index(i+2,j+2,k,0);
     int p571=mcL[l]*mcL[l];
     l=index(i+2,j-1,k,0);
     int p580=mcL[l];
     l=index(i+2,j-1,k,0);
     int p581=mcL[l]*mcL[l];
     l=index(i+2,j,k+2,0);
     int p590=mcL[l];
     l=index(i+2,j,k+2,0);
     int p591=mcL[l]*mcL[l];
     l=index(i+2,j,k-1,0);
     int p600=mcL[l];
     l=index(i+2,j,k-1,0);
     int p601=mcL[l]*mcL[l];
     l=index(i-1,j+2,k+2,0);
     int p610=mcL[l];
     l=index(i-1,j+2,k+2,0);
     int p611=mcL[l]*mcL[l];
     l=index(i-1,j+2,k,0);
     int p620=mcL[l];
     l=index(i-1,j+2,k,0);
     int p621=mcL[l]*mcL[l];
     l=index(i-1,j,k+2,0);
     int p630=mcL[l];
     l=index(i-1,j,k+2,0);
     int p631=mcL[l]*mcL[l];
     l=index(i,j+2,k+2,0);
     int p640=mcL[l];
     l=index(i,j+2,k+2,0);
     int p641=mcL[l]*mcL[l];
     l=index(i,j+2,k-1,0);
     int p650=mcL[l];
     l=index(i,j+2,k-1,0);
     int p651=mcL[l]*mcL[l];
     l=index(i,j-1,k+2,0);
     int p660=mcL[l];
     l=index(i,j-1,k+2,0);
     int p661=mcL[l]*mcL[l];

     energy = energy-6.9586*(p01)+0.658756*(p00)-0.0173448*(p01*p10+p01*p20+p01*p30+p01*p40)-0.159011*(p00*p11+p00*p21+p00*p31+p00*p41)+0.679524*(p01*p11+p01*p21+p01*p31+p01*p41)+0.0841386*(p00*p10+p00*p20+p00*p30+p00*p40)-0.092346*(p51*p00+p01*p50+p61*p00+p01*p60+p01*p70+p71*p00+p01*p80+p81*p00+p01*p90+p91*p00+p101*p00+p01*p100)+0.275366*(p51*p01+p61*p01+p01*p71+p01*p81+p01*p91+p101*p01)+0.0551302*(p111*p01+p01*p121+p131*p01+p141*p01)-0.0138053*(p110*p00+p00*p120+p130*p00+p140*p00)-0.00161467*(p01*p151+p01*p161+p01*p171+p01*p181+p01*p191+p01*p201+p01*p211+p01*p221+p01*p231+p01*p241+p01*p251+p01*p261)-0.0383274*(p01*p270+p01*p280+p01*p290+p01*p300)+0.00311609*(p01*p271+p01*p281+p01*p291+p01*p301)+0.0421336*(p00*p270+p00*p280+p00*p290+p00*p300)+0.00812835*(p01*p310+p321*p00+p01*p320+p311*p00+p01*p330+p341*p00+p01*p340+p331*p00+p01*p350+p361*p00+p01*p360+p351*p00+p01*p370+p381*p00+p01*p380+p371*p00+p01*p390+p401*p00+p01*p400+p391*p00+p01*p410+p421*p00+p01*p420+p411*p00)+0.00956795*(p00*p310+p320*p00+p00*p330+p340*p00+p00*p350+p360*p00+p00*p370+p380*p00+p00*p390+p400*p00+p00*p410+p420*p00)-0.00450773*(p430*p440*p01+p450*p460*p01+p470*p480*p01+p490*p500*p01+p510*p520*p01+p530*p540*p01)-0.00212345*(p430*p440*p00+p450*p460*p00+p470*p480*p00+p490*p500*p00+p510*p520*p00+p530*p540*p00)-0.00158958*(p00*p40*p531+p00*p40*p491+p00*p30*p541+p00*p30*p511+p00*p20*p501+p00*p20*p521+p00*p40*p431+p00*p30*p451+p00*p20*p471+p00*p10*p441+p00*p10*p461+p00*p10*p481)-3.18114e-05*(p251*p200*p01+p201*p250*p01+p241*p180*p01+p181*p240*p01+p261*p210*p01+p211*p260*p01+p241*p190*p01+p191*p240*p01+p261*p230*p01+p231*p260*p01+p251*p220*p01+p221*p250*p01+p221*p200*p01+p201*p220*p01+p171*p150*p01+p151*p170*p01+p231*p210*p01+p211*p230*p01+p171*p160*p01+p161*p170*p01+p191*p180*p01+p181*p190*p01+p161*p150*p01+p151*p160*p01)+0.00780764*(p00*p471*p240+p00*p451*p250+p00*p431*p260+p00*p481*p170+p00*p511*p220+p00*p491*p230+p00*p461*p160+p00*p521*p190+p00*p441*p150+p00*p501*p180+p00*p531*p210+p00*p541*p200)-0.000457364*(p01*p40*p160+p01*p40*p170+p01*p30*p150+p01*p30*p170+p01*p20*p150+p01*p20*p160+p01*p40*p190+p01*p40*p240+p01*p30*p180+p01*p30*p240+p01*p40*p220+p01*p40*p250+p01*p30*p230+p01*p30*p260+p01*p20*p200+p01*p20*p250+p01*p20*p210+p01*p20*p260+p01*p10*p180+p01*p10*p190+p01*p10*p200+p01*p10*p220+p01*p10*p210+p01*p10*p230)+0.00433732*(p01*p470*p140+p01*p450*p130+p131*p550*p00+p141*p560*p00+p01*p430*p120+p121*p570*p00+p141*p580*p00+p121*p590*p00+p131*p600*p00+p01*p480*p110+p111*p610*p00+p01*p510*p130+p111*p620*p00+p01*p490*p120+p01*p460*p110+p01*p520*p140+p01*p440*p110+p01*p500*p140+p111*p630*p00+p01*p530*p120+p01*p540*p130+p121*p640*p00+p131*p650*p00+p141*p660*p00)+0.00518956*(p00*p471*p140+p00*p451*p130+p130*p551*p00+p140*p561*p00+p00*p431*p120+p120*p571*p00+p140*p581*p00+p120*p591*p00+p130*p601*p00+p00*p481*p110+p110*p611*p00+p00*p511*p130+p110*p621*p00+p00*p491*p120+p00*p461*p110+p00*p521*p140+p00*p441*p110+p00*p501*p140+p110*p631*p00+p00*p531*p120+p00*p541*p130+p120*p641*p00+p130*p651*p00+p140*p661*p00)+0.0118966*(p00*p470*p141+p00*p450*p131+p130*p550*p01+p140*p560*p01+p00*p430*p121+p120*p570*p01+p140*p580*p01+p120*p590*p01+p130*p600*p01+p00*p480*p111+p110*p610*p01+p00*p510*p131+p110*p620*p01+p00*p490*p121+p00*p460*p111+p00*p520*p141+p00*p440*p111+p00*p500*p141+p110*p630*p01+p00*p530*p121+p00*p540*p131+p120*p640*p01+p130*p650*p01+p140*p660*p01)+0.00789307*(p221*p550*p00+p191*p560*p00+p231*p570*p00+p181*p580*p00+p211*p590*p00+p201*p600*p00+p251*p550*p00+p161*p610*p00+p261*p570*p00+p151*p620*p00+p241*p560*p00+p171*p610*p00+p241*p580*p00+p171*p620*p00+p261*p590*p00+p151*p630*p00+p251*p600*p00+p161*p630*p00+p211*p640*p00+p201*p650*p00+p231*p640*p00+p181*p660*p00+p221*p650*p00+p191*p660*p00)+0.0011537*(p220*p551*p00+p190*p561*p00+p230*p571*p00+p180*p581*p00+p210*p591*p00+p200*p601*p00+p250*p551*p00+p160*p611*p00+p260*p571*p00+p150*p621*p00+p240*p561*p00+p170*p611*p00+p240*p581*p00+p170*p621*p00+p260*p591*p00+p150*p631*p00+p250*p601*p00+p160*p631*p00+p210*p641*p00+p200*p651*p00+p230*p641*p00+p180*p661*p00+p220*p651*p00+p190*p661*p00)+0.00403165*(p00*p241*p260+p00*p261*p240+p00*p251*p260+p00*p261*p250+p00*p241*p250+p00*p251*p240+p00*p171*p230+p00*p231*p170+p00*p221*p230+p00*p231*p220+p00*p171*p220+p00*p221*p170+p00*p161*p210+p00*p211*p160+p00*p191*p210+p00*p211*p190+p00*p151*p200+p00*p201*p150+p00*p181*p200+p00*p201*p180+p00*p161*p190+p00*p191*p160+p00*p151*p180+p00*p181*p150);
     return energy;
  }


}



 
//************************************************************ 
 
void Monte_Carlo::pointcorr(int i, int j, int k, int b){
  int l; 
  if(b == 0){
     l=index(i,j,k,0); 
     double p0=mcL[l]; 
     l=index(i,j-1,k-1,2); 
     double p1=mcL[l]; 
     l=index(i,j,k,1); 
     double p2=mcL[l]; 
     l=index(i,j,k-1,1); 
     double p3=mcL[l]; 
     l=index(i,j-1,k,1); 
     double p4=mcL[l]; 
     l=index(i-1,j,k-1,2); 
     double p5=mcL[l]; 
     l=index(i-1,j-1,k,2); 
     double p6=mcL[l]; 
     l=index(i-1,j-1,k-1,2); 
     double p7=mcL[l]; 
     l=index(i-1,j,k,1); 
     double p8=mcL[l]; 
     l=index(i+1,j,k,0); 
     double p9=mcL[l]; 
     l=index(i-1,j,k,0); 
     double p10=mcL[l]; 
     l=index(i+1,j,k-1,0); 
     double p11=mcL[l]; 
     l=index(i-1,j,k+1,0); 
     double p12=mcL[l]; 
     l=index(i+1,j-1,k,0); 
     double p13=mcL[l]; 
     l=index(i-1,j+1,k,0); 
     double p14=mcL[l]; 
     l=index(i,j+1,k,0); 
     double p15=mcL[l]; 
     l=index(i,j-1,k,0); 
     double p16=mcL[l]; 
     l=index(i,j+1,k-1,0); 
     double p17=mcL[l]; 
     l=index(i,j-1,k+1,0); 
     double p18=mcL[l]; 
     l=index(i,j,k+1,0); 
     double p19=mcL[l]; 
     l=index(i,j,k-1,0); 
     double p20=mcL[l]; 
     l=index(i+1,j+1,k-1,0); 
     double p21=mcL[l]; 
     l=index(i-1,j-1,k+1,0); 
     double p22=mcL[l]; 
     l=index(i+1,j-1,k+1,0); 
     double p23=mcL[l]; 
     l=index(i-1,j+1,k-1,0); 
     double p24=mcL[l]; 
     l=index(i-1,j+1,k+1,0); 
     double p25=mcL[l]; 
     l=index(i+1,j-1,k-1,0); 
     double p26=mcL[l]; 
     l=index(i+1,j-1,k-1,2); 
     double p27=mcL[l]; 
     l=index(i+1,j,k,1); 
     double p28=mcL[l]; 
     l=index(i+1,j-1,k-2,2); 
     double p29=mcL[l]; 
     l=index(i+1,j,k-2,1); 
     double p30=mcL[l]; 
     l=index(i+1,j-2,k-1,2); 
     double p31=mcL[l]; 
     l=index(i+1,j-2,k,1); 
     double p32=mcL[l]; 
     l=index(i,j+1,k,1); 
     double p33=mcL[l]; 
     l=index(i-1,j+1,k-1,2); 
     double p34=mcL[l]; 
     l=index(i,j+1,k-2,1); 
     double p35=mcL[l]; 
     l=index(i-1,j+1,k-2,2); 
     double p36=mcL[l]; 
     l=index(i,j,k+1,1); 
     double p37=mcL[l]; 
     l=index(i-1,j-1,k+1,2); 
     double p38=mcL[l]; 
     l=index(i,j,k-2,1); 
     double p39=mcL[l]; 
     l=index(i-1,j-1,k-2,2); 
     double p40=mcL[l]; 
     l=index(i,j-2,k+1,1); 
     double p41=mcL[l]; 
     l=index(i-1,j-2,k+1,2); 
     double p42=mcL[l]; 
     l=index(i,j-2,k,1); 
     double p43=mcL[l]; 
     l=index(i-1,j-2,k-1,2); 
     double p44=mcL[l]; 
     l=index(i-2,j+1,k-1,2); 
     double p45=mcL[l]; 
     l=index(i-2,j+1,k,1); 
     double p46=mcL[l]; 
     l=index(i-2,j-1,k+1,2); 
     double p47=mcL[l]; 
     l=index(i-2,j,k+1,1); 
     double p48=mcL[l]; 
     l=index(i-2,j-1,k-1,2); 
     double p49=mcL[l]; 
     l=index(i-2,j,k,1); 
     double p50=mcL[l]; 
     l=index(i+2,j,k-1,0); 
     double p51=mcL[l]; 
     l=index(i-2,j,k+1,0); 
     double p52=mcL[l]; 
     l=index(i+2,j-1,k,0); 
     double p53=mcL[l]; 
     l=index(i-2,j+1,k,0); 
     double p54=mcL[l]; 
     l=index(i+1,j+1,k,0); 
     double p55=mcL[l]; 
     l=index(i-1,j-1,k,0); 
     double p56=mcL[l]; 
     l=index(i+1,j,k+1,0); 
     double p57=mcL[l]; 
     l=index(i-1,j,k-1,0); 
     double p58=mcL[l]; 
     l=index(i+2,j-1,k-1,0); 
     double p59=mcL[l]; 
     l=index(i-2,j+1,k+1,0); 
     double p60=mcL[l]; 
     l=index(i+1,j+1,k-2,0); 
     double p61=mcL[l]; 
     l=index(i-1,j-1,k+2,0); 
     double p62=mcL[l]; 
     l=index(i-1,j,k+2,0); 
     double p63=mcL[l]; 
     l=index(i+1,j,k-2,0); 
     double p64=mcL[l]; 
     l=index(i+1,j-2,k+1,0); 
     double p65=mcL[l]; 
     l=index(i-1,j+2,k-1,0); 
     double p66=mcL[l]; 
     l=index(i-1,j+2,k,0); 
     double p67=mcL[l]; 
     l=index(i+1,j-2,k,0); 
     double p68=mcL[l]; 
     l=index(i,j+2,k-1,0); 
     double p69=mcL[l]; 
     l=index(i,j-2,k+1,0); 
     double p70=mcL[l]; 
     l=index(i,j+1,k+1,0); 
     double p71=mcL[l]; 
     l=index(i,j-1,k-1,0); 
     double p72=mcL[l]; 
     l=index(i,j+1,k-2,0); 
     double p73=mcL[l]; 
     l=index(i,j-1,k+2,0); 
     double p74=mcL[l]; 
     l=index(i,j,k,2); 
     double p75=mcL[l]; 
     l=index(i+2,j-1,k-1,1); 
     double p76=mcL[l]; 
     l=index(i,j,k-3,2); 
     double p77=mcL[l]; 
     l=index(i,j-3,k,2); 
     double p78=mcL[l]; 
     l=index(i-1,j+2,k-1,1); 
     double p79=mcL[l]; 
     l=index(i-1,j-1,k+2,1); 
     double p80=mcL[l]; 
     l=index(i-1,j-1,k-1,1); 
     double p81=mcL[l]; 
     l=index(i-3,j,k,2); 
     double p82=mcL[l]; 
     l=index(i+2,j,k,0); 
     double p83=mcL[l]; 
     l=index(i-2,j,k,0); 
     double p84=mcL[l]; 
     l=index(i+2,j,k-2,0); 
     double p85=mcL[l]; 
     l=index(i-2,j,k+2,0); 
     double p86=mcL[l]; 
     l=index(i+2,j-2,k,0); 
     double p87=mcL[l]; 
     l=index(i-2,j+2,k,0); 
     double p88=mcL[l]; 
     l=index(i,j+2,k,0); 
     double p89=mcL[l]; 
     l=index(i,j-2,k,0); 
     double p90=mcL[l]; 
     l=index(i,j+2,k-2,0); 
     double p91=mcL[l]; 
     l=index(i,j-2,k+2,0); 
     double p92=mcL[l]; 
     l=index(i,j,k+2,0); 
     double p93=mcL[l]; 
     l=index(i,j,k-2,0); 
     double p94=mcL[l]; 
     l=index(i-2,j,k,2); 
     double p95=mcL[l]; 
     l=index(i-1,j,k,2); 
     double p96=mcL[l]; 
     l=index(i,j-1,k-1,1); 
     double p97=mcL[l]; 
     l=index(i+1,j-1,k-1,1); 
     double p98=mcL[l]; 
     l=index(i-2,j,k-1,2); 
     double p99=mcL[l]; 
     l=index(i-1,j,k-2,2); 
     double p100=mcL[l]; 
     l=index(i,j-1,k+1,1); 
     double p101=mcL[l]; 
     l=index(i+1,j-1,k,1); 
     double p102=mcL[l]; 
     l=index(i-2,j-1,k,2); 
     double p103=mcL[l]; 
     l=index(i-1,j-2,k,2); 
     double p104=mcL[l]; 
     l=index(i,j+1,k-1,1); 
     double p105=mcL[l]; 
     l=index(i+1,j,k-1,1); 
     double p106=mcL[l]; 
     l=index(i-1,j,k-1,1); 
     double p107=mcL[l]; 
     l=index(i-1,j+1,k-1,1); 
     double p108=mcL[l]; 
     l=index(i,j-2,k,2); 
     double p109=mcL[l]; 
     l=index(i,j-1,k,2); 
     double p110=mcL[l]; 
     l=index(i-1,j,k+1,1); 
     double p111=mcL[l]; 
     l=index(i-1,j+1,k,1); 
     double p112=mcL[l]; 
     l=index(i,j-2,k-1,2); 
     double p113=mcL[l]; 
     l=index(i,j-1,k-2,2); 
     double p114=mcL[l]; 
     l=index(i-1,j-1,k,1); 
     double p115=mcL[l]; 
     l=index(i-1,j-1,k+1,1); 
     double p116=mcL[l]; 
     l=index(i,j,k-2,2); 
     double p117=mcL[l]; 
     l=index(i,j,k-1,2); 
     double p118=mcL[l]; 
     l=index(i-1,j-2,k+1,1); 
     double p119=mcL[l]; 
     l=index(i-1,j+1,k-2,1); 
     double p120=mcL[l]; 
     l=index(i-2,j-2,k+1,2); 
     double p121=mcL[l]; 
     l=index(i-2,j+1,k-2,2); 
     double p122=mcL[l]; 
     l=index(i-1,j+1,k+1,1); 
     double p123=mcL[l]; 
     l=index(i-2,j-2,k,2); 
     double p124=mcL[l]; 
     l=index(i-2,j+1,k,2); 
     double p125=mcL[l]; 
     l=index(i-2,j,k-2,2); 
     double p126=mcL[l]; 
     l=index(i-2,j,k+1,2); 
     double p127=mcL[l]; 
     l=index(i-2,j-1,k+1,1); 
     double p128=mcL[l]; 
     l=index(i+1,j-2,k-2,2); 
     double p129=mcL[l]; 
     l=index(i+1,j-1,k-2,1); 
     double p130=mcL[l]; 
     l=index(i+1,j-2,k,2); 
     double p131=mcL[l]; 
     l=index(i+1,j-1,k+1,1); 
     double p132=mcL[l]; 
     l=index(i-2,j+1,k-1,1); 
     double p133=mcL[l]; 
     l=index(i+1,j-2,k-1,1); 
     double p134=mcL[l]; 
     l=index(i-2,j+1,k+1,1); 
     double p135=mcL[l]; 
     l=index(i+1,j-2,k+1,1); 
     double p136=mcL[l]; 
     l=index(i+1,j,k-2,2); 
     double p137=mcL[l]; 
     l=index(i+1,j+1,k-1,1); 
     double p138=mcL[l]; 
     l=index(i+1,j+1,k-2,1); 
     double p139=mcL[l]; 
     l=index(i,j-2,k-2,2); 
     double p140=mcL[l]; 
     l=index(i,j-2,k+1,2); 
     double p141=mcL[l]; 
     l=index(i,j+1,k-2,2); 
     double p142=mcL[l]; 

     AVcorr[0]+=1.0/3;
     AVcorr[3]+=(p0*p0);
     AVcorr[4]+=(p0);
     AVcorr[5]+=(p1*p1*p0+p2*p2*p0+p3*p3*p0+p4*p4*p0+p5*p5*p0+p6*p6*p0+p7*p7*p0+p8*p8*p0)/16;
     AVcorr[6]+=(p1*p0*p0+p2*p0*p0+p3*p0*p0+p4*p0*p0+p5*p0*p0+p6*p0*p0+p7*p0*p0+p8*p0*p0)/16;
     AVcorr[7]+=(p1*p1*p0*p0+p2*p2*p0*p0+p3*p3*p0*p0+p4*p4*p0*p0+p5*p5*p0*p0+p6*p6*p0*p0+p7*p7*p0*p0+p8*p8*p0*p0)/16;
     AVcorr[8]+=(p1*p0+p2*p0+p3*p0+p4*p0+p5*p0+p6*p0+p7*p0+p8*p0)/16;
     AVcorr[15]+=(p0*p0*p9+p10*p10*p0+p0*p0*p10+p9*p9*p0+p0*p0*p11+p12*p12*p0+p0*p0*p12+p11*p11*p0+p0*p0*p13+p14*p14*p0+p0*p0*p14+p13*p13*p0+p0*p0*p15+p16*p16*p0+p0*p0*p16+p15*p15*p0+p0*p0*p17+p18*p18*p0+p0*p0*p18+p17*p17*p0+p0*p0*p19+p20*p20*p0+p0*p0*p20+p19*p19*p0)/24;
     AVcorr[28]+=(p0*p0*p21+p22*p22*p0+p0*p0*p22+p21*p21*p0+p0*p0*p23+p24*p24*p0+p0*p0*p24+p23*p23*p0+p0*p0*p25+p26*p26*p0+p0*p0*p26+p25*p25*p0)/12;
     AVcorr[36]+=(p27*p27*p0*p0+p28*p28*p0*p0+p29*p29*p0*p0+p30*p30*p0*p0+p31*p31*p0*p0+p32*p32*p0*p0+p33*p33*p0*p0+p34*p34*p0*p0+p35*p35*p0*p0+p36*p36*p0*p0+p37*p37*p0*p0+p38*p38*p0*p0+p39*p39*p0*p0+p40*p40*p0*p0+p41*p41*p0*p0+p42*p42*p0*p0+p43*p43*p0*p0+p44*p44*p0*p0+p45*p45*p0*p0+p46*p46*p0*p0+p47*p47*p0*p0+p48*p48*p0*p0+p49*p49*p0*p0+p50*p50*p0*p0)/48;
     AVcorr[46]+=(p0*p0*p51*p51+p52*p52*p0*p0+p0*p0*p53*p53+p54*p54*p0*p0+p0*p0*p55*p55+p56*p56*p0*p0+p0*p0*p57*p57+p58*p58*p0*p0+p0*p0*p59*p59+p60*p60*p0*p0+p0*p0*p61*p61+p62*p62*p0*p0+p0*p0*p63*p63+p64*p64*p0*p0+p0*p0*p65*p65+p66*p66*p0*p0+p0*p0*p67*p67+p68*p68*p0*p0+p0*p0*p69*p69+p70*p70*p0*p0+p0*p0*p71*p71+p72*p72*p0*p0+p0*p0*p73*p73+p74*p74*p0*p0)/24;
     AVcorr[47]+=(p0*p51+p52*p0+p0*p53+p54*p0+p0*p55+p56*p0+p0*p57+p58*p0+p0*p59+p60*p0+p0*p61+p62*p0+p0*p63+p64*p0+p0*p65+p66*p0+p0*p67+p68*p0+p0*p69+p70*p0+p0*p71+p72*p0+p0*p73+p74*p0)/24;
     AVcorr[48]+=(p75*p75*p0+p76*p76*p0+p77*p77*p0+p78*p78*p0+p79*p79*p0+p80*p80*p0+p81*p81*p0+p82*p82*p0)/16;
     AVcorr[50]+=(p75*p75*p0*p0+p76*p76*p0*p0+p77*p77*p0*p0+p78*p78*p0*p0+p79*p79*p0*p0+p80*p80*p0*p0+p81*p81*p0*p0+p82*p82*p0*p0)/16;
     AVcorr[51]+=(p75*p0+p76*p0+p77*p0+p78*p0+p79*p0+p80*p0+p81*p0+p82*p0)/16;
     AVcorr[58]+=(p0*p83+p84*p0+p0*p85+p86*p0+p0*p87+p88*p0+p0*p89+p90*p0+p0*p91+p92*p0+p0*p93+p94*p0)/12;
     AVcorr[86]+=(p0*p0*p10*p12+p9*p9*p0*p19+p11*p11*p20*p0+p0*p0*p20*p11+p19*p19*p0*p9+p12*p12*p10*p0+p0*p0*p19*p9+p20*p20*p0*p11+p10*p10*p12*p0+p0*p0*p10*p14+p9*p9*p0*p15+p13*p13*p16*p0+p0*p0*p16*p13+p15*p15*p0*p9+p14*p14*p10*p0+p0*p0*p15*p9+p16*p16*p0*p13+p10*p10*p14*p0+p0*p0*p10*p16+p9*p9*p0*p13+p15*p15*p14*p0+p0*p0*p14*p15+p13*p13*p0*p9+p16*p16*p10*p0+p0*p0*p13*p9+p14*p14*p0*p15+p10*p10*p16*p0+p0*p0*p10*p20+p9*p9*p0*p11+p19*p19*p12*p0+p0*p0*p12*p19+p11*p11*p0*p9+p20*p20*p10*p0+p0*p0*p11*p9+p12*p12*p0*p19+p10*p10*p20*p0+p0*p0*p12*p14+p11*p11*p0*p17+p13*p13*p18*p0+p0*p0*p18*p13+p17*p17*p0*p11+p14*p14*p12*p0+p0*p0*p17*p11+p18*p18*p0*p13+p12*p12*p14*p0+p0*p0*p12*p18+p11*p11*p0*p13+p17*p17*p14*p0+p0*p0*p14*p17+p13*p13*p0*p11+p18*p18*p12*p0+p0*p0*p13*p11+p14*p14*p0*p17+p12*p12*p18*p0+p0*p0*p16*p18+p15*p15*p0*p19+p17*p17*p20*p0+p0*p0*p20*p17+p19*p19*p0*p15+p18*p18*p16*p0+p0*p0*p19*p15+p20*p20*p0*p17+p16*p16*p18*p0+p0*p0*p16*p20+p15*p15*p0*p17+p19*p19*p18*p0+p0*p0*p18*p19+p17*p17*p0*p15+p20*p20*p16*p0+p0*p0*p17*p15+p18*p18*p0*p19+p16*p16*p20*p0)/72;
     AVcorr[89]+=(p0*p10*p12+p9*p0*p19+p11*p20*p0+p0*p10*p14+p9*p0*p15+p13*p16*p0+p0*p10*p16+p9*p0*p13+p15*p14*p0+p0*p10*p20+p9*p0*p11+p19*p12*p0+p0*p12*p14+p11*p0*p17+p13*p18*p0+p0*p12*p18+p11*p0*p13+p17*p14*p0+p0*p16*p18+p15*p0*p19+p17*p20*p0+p0*p16*p20+p15*p0*p17+p19*p18*p0)/24;
     AVcorr[134]+=(p0*p10*p95*p95+p9*p0*p96*p96+p0*p10*p97*p97+p9*p0*p98*p98+p0*p12*p99*p99+p11*p0*p100*p100+p0*p12*p101*p101+p11*p0*p102*p102+p0*p14*p103*p103+p13*p0*p104*p104+p0*p14*p105*p105+p13*p0*p106*p106+p0*p16*p107*p107+p15*p0*p108*p108+p0*p16*p109*p109+p15*p0*p110*p110+p0*p18*p111*p111+p17*p0*p112*p112+p0*p18*p113*p113+p17*p0*p114*p114+p0*p20*p115*p115+p19*p0*p116*p116+p0*p20*p117*p117+p19*p0*p118*p118)/36;
     AVcorr[137]+=(p0*p10*p95+p9*p0*p96+p0*p10*p97+p9*p0*p98+p0*p12*p99+p11*p0*p100+p0*p12*p101+p11*p0*p102+p0*p14*p103+p13*p0*p104+p0*p14*p105+p13*p0*p106+p0*p16*p107+p15*p0*p108+p0*p16*p109+p15*p0*p110+p0*p18*p111+p17*p0*p112+p0*p18*p113+p17*p0*p114+p0*p20*p115+p19*p0*p116+p0*p20*p117+p19*p0*p118)/36;
     AVcorr[197]+=(p7*p0*p22*p22+p117*p21*p0*p0+p7*p0*p24*p24+p109*p23*p0*p0+p8*p0*p22*p22+p105*p21*p0*p0+p8*p0*p24*p24+p101*p23*p0*p0+p6*p0*p22*p22+p118*p21*p0*p0+p6*p0*p25*p25+p113*p26*p0*p0+p8*p0*p25*p25+p97*p26*p0*p0+p5*p0*p24*p24+p110*p23*p0*p0+p5*p0*p25*p25+p114*p26*p0*p0+p4*p0*p22*p22+p106*p21*p0*p0+p4*p0*p26*p26+p111*p25*p0*p0+p7*p0*p26*p26+p95*p25*p0*p0+p4*p0*p23*p23+p107*p24*p0*p0+p6*p0*p23*p23+p99*p24*p0*p0+p3*p0*p24*p24+p102*p23*p0*p0+p3*p0*p26*p26+p112*p25*p0*p0+p2*p0*p25*p25+p98*p26*p0*p0+p2*p0*p23*p23+p108*p24*p0*p0+p3*p0*p21*p21+p115*p22*p0*p0+p5*p0*p21*p21+p103*p22*p0*p0+p2*p0*p21*p21+p116*p22*p0*p0+p1*p0*p26*p26+p96*p25*p0*p0+p1*p0*p23*p23+p100*p24*p0*p0+p1*p0*p21*p21+p104*p22*p0*p0)/72;
     AVcorr[202]+=(p0*p0*p18*p9+p17*p17*p0*p21+p10*p10*p22*p0+p0*p0*p17*p9+p18*p18*p0*p23+p10*p10*p24*p0+p0*p0*p16*p11+p15*p15*p0*p21+p12*p12*p22*p0+p0*p0*p11*p15+p12*p12*p0*p25+p16*p16*p26*p0+p0*p0*p20*p13+p19*p19*p0*p23+p14*p14*p24*p0+p0*p0*p13*p19+p14*p14*p0*p25+p20*p20*p26*p0+p0*p0*p12*p15+p11*p11*p0*p21+p16*p16*p22*p0+p0*p0*p10*p17+p9*p9*p0*p21+p18*p18*p22*p0+p0*p0*p14*p19+p13*p13*p0*p23+p20*p20*p24*p0+p0*p0*p20*p14+p19*p19*p0*p25+p13*p13*p26*p0+p0*p0*p10*p18+p9*p9*p0*p23+p17*p17*p24*p0+p0*p0*p16*p12+p15*p15*p0*p25+p11*p11*p26*p0)/36;
     AVcorr[206]+=(p0*p0*p18*p18*p9*p9+p17*p17*p0*p0*p21*p21+p10*p10*p22*p22*p0*p0+p0*p0*p17*p17*p9*p9+p18*p18*p0*p0*p23*p23+p10*p10*p24*p24*p0*p0+p0*p0*p16*p16*p11*p11+p15*p15*p0*p0*p21*p21+p12*p12*p22*p22*p0*p0+p0*p0*p11*p11*p15*p15+p12*p12*p0*p0*p25*p25+p16*p16*p26*p26*p0*p0+p0*p0*p20*p20*p13*p13+p19*p19*p0*p0*p23*p23+p14*p14*p24*p24*p0*p0+p0*p0*p13*p13*p19*p19+p14*p14*p0*p0*p25*p25+p20*p20*p26*p26*p0*p0+p0*p0*p12*p12*p15*p15+p11*p11*p0*p0*p21*p21+p16*p16*p22*p22*p0*p0+p0*p0*p10*p10*p17*p17+p9*p9*p0*p0*p21*p21+p18*p18*p22*p22*p0*p0+p0*p0*p14*p14*p19*p19+p13*p13*p0*p0*p23*p23+p20*p20*p24*p24*p0*p0+p0*p0*p20*p20*p14*p14+p19*p19*p0*p0*p25*p25+p13*p13*p26*p26*p0*p0+p0*p0*p10*p10*p18*p18+p9*p9*p0*p0*p23*p23+p17*p17*p24*p24*p0*p0+p0*p0*p16*p16*p12*p12+p15*p15*p0*p0*p25*p25+p11*p11*p26*p26*p0*p0)/36;
     AVcorr[207]+=(p0*p18*p9+p17*p0*p21+p10*p22*p0+p0*p17*p9+p18*p0*p23+p10*p24*p0+p0*p16*p11+p15*p0*p21+p12*p22*p0+p0*p11*p15+p12*p0*p25+p16*p26*p0+p0*p20*p13+p19*p0*p23+p14*p24*p0+p0*p13*p19+p14*p0*p25+p20*p26*p0+p0*p12*p15+p11*p0*p21+p16*p22*p0+p0*p10*p17+p9*p0*p21+p18*p22*p0+p0*p14*p19+p13*p0*p23+p20*p24*p0+p0*p20*p14+p19*p0*p25+p13*p26*p0+p0*p10*p18+p9*p0*p23+p17*p24*p0+p0*p16*p12+p15*p0*p25+p11*p26*p0)/36;
     AVcorr[301]+=(p0*p0*p10*p47*p47+p9*p9*p0*p38*p38+p0*p0*p9*p38*p38+p10*p10*p0*p47*p47+p0*p0*p10*p45*p45+p9*p9*p0*p34*p34+p0*p0*p9*p34*p34+p10*p10*p0*p45*p45+p0*p0*p10*p43*p43+p9*p9*p0*p32*p32+p0*p0*p9*p32*p32+p10*p10*p0*p43*p43+p0*p0*p10*p39*p39+p9*p9*p0*p30*p30+p0*p0*p9*p30*p30+p10*p10*p0*p39*p39+p0*p0*p12*p49*p49+p11*p11*p0*p40*p40+p0*p0*p11*p40*p40+p12*p12*p0*p49*p49+p0*p0*p12*p45*p45+p11*p11*p0*p36*p36+p0*p0*p11*p36*p36+p12*p12*p0*p45*p45+p0*p0*p12*p41*p41+p11*p11*p0*p32*p32+p0*p0*p11*p32*p32+p12*p12*p0*p41*p41+p0*p0*p12*p37*p37+p11*p11*p0*p28*p28+p0*p0*p11*p28*p28+p12*p12*p0*p37*p37+p0*p0*p14*p49*p49+p13*p13*p0*p44*p44+p0*p0*p13*p44*p44+p14*p14*p0*p49*p49+p0*p0*p14*p47*p47+p13*p13*p0*p42*p42+p0*p0*p13*p42*p42+p14*p14*p0*p47*p47+p0*p0*p14*p35*p35+p13*p13*p0*p30*p30+p0*p0*p13*p30*p30+p14*p14*p0*p35*p35+p0*p0*p14*p33*p33+p13*p13*p0*p28*p28+p0*p0*p13*p28*p28+p14*p14*p0*p33*p33+p0*p0*p16*p50*p50+p15*p15*p0*p46*p46+p0*p0*p15*p46*p46+p16*p16*p0*p50*p50+p0*p0*p16*p42*p42+p15*p15*p0*p38*p38+p0*p0*p15*p38*p38+p16*p16*p0*p42*p42+p0*p0*p16*p39*p39+p15*p15*p0*p35*p35+p0*p0*p15*p35*p35+p16*p16*p0*p39*p39+p0*p0*p16*p31*p31+p15*p15*p0*p27*p27+p0*p0*p15*p27*p27+p16*p16*p0*p31*p31+p0*p0*p18*p48*p48+p17*p17*p0*p46*p46+p0*p0*p17*p46*p46+p18*p18*p0*p48*p48+p0*p0*p18*p44*p44+p17*p17*p0*p40*p40+p0*p0*p17*p40*p40+p18*p18*p0*p44*p44+p0*p0*p18*p37*p37+p17*p17*p0*p33*p33+p0*p0*p17*p33*p33+p18*p18*p0*p37*p37+p0*p0*p18*p31*p31+p17*p17*p0*p29*p29+p0*p0*p17*p29*p29+p18*p18*p0*p31*p31+p0*p0*p20*p50*p50+p19*p19*p0*p48*p48+p0*p0*p19*p48*p48+p20*p20*p0*p50*p50+p0*p0*p20*p43*p43+p19*p19*p0*p41*p41+p0*p0*p19*p41*p41+p20*p20*p0*p43*p43+p0*p0*p20*p36*p36+p19*p19*p0*p34*p34+p0*p0*p19*p34*p34+p20*p20*p0*p36*p36+p0*p0*p20*p29*p29+p19*p19*p0*p27*p27+p0*p0*p19*p27*p27+p20*p20*p0*p29*p29)/144;
     AVcorr[529]+=(p0*p12*p12*p9*p9+p11*p0*p0*p51*p51+p10*p52*p52*p0*p0+p0*p14*p14*p9*p9+p13*p0*p0*p53*p53+p10*p54*p54*p0*p0+p0*p16*p16*p9*p9+p15*p0*p0*p55*p55+p10*p56*p56*p0*p0+p0*p20*p20*p9*p9+p19*p0*p0*p57*p57+p10*p58*p58*p0*p0+p0*p10*p10*p11*p11+p9*p0*p0*p51*p51+p12*p52*p52*p0*p0+p0*p14*p14*p11*p11+p13*p0*p0*p59*p59+p12*p60*p60*p0*p0+p0*p18*p18*p11*p11+p17*p0*p0*p61*p61+p12*p62*p62*p0*p0+p0*p11*p11*p19*p19+p12*p0*p0*p63*p63+p20*p64*p64*p0*p0+p0*p10*p10*p13*p13+p9*p0*p0*p53*p53+p14*p54*p54*p0*p0+p0*p12*p12*p13*p13+p11*p0*p0*p59*p59+p14*p60*p60*p0*p0+p0*p17*p17*p13*p13+p18*p0*p0*p65*p65+p14*p66*p66*p0*p0+p0*p13*p13*p15*p15+p14*p0*p0*p67*p67+p16*p68*p68*p0*p0+p0*p10*p10*p15*p15+p9*p0*p0*p55*p55+p16*p56*p56*p0*p0+p0*p18*p18*p15*p15+p17*p0*p0*p69*p69+p16*p70*p70*p0*p0+p0*p20*p20*p15*p15+p19*p0*p0*p71*p71+p16*p72*p72*p0*p0+p0*p12*p12*p17*p17+p11*p0*p0*p61*p61+p18*p62*p62*p0*p0+p0*p16*p16*p17*p17+p15*p0*p0*p69*p69+p18*p70*p70*p0*p0+p0*p19*p19*p17*p17+p20*p0*p0*p73*p73+p18*p74*p74*p0*p0+p0*p10*p10*p19*p19+p9*p0*p0*p57*p57+p20*p58*p58*p0*p0+p0*p16*p16*p19*p19+p15*p0*p0*p71*p71+p20*p72*p72*p0*p0+p0*p20*p20*p12*p12+p19*p0*p0*p63*p63+p11*p64*p64*p0*p0+p0*p18*p18*p20*p20+p17*p0*p0*p73*p73+p19*p74*p74*p0*p0+p0*p14*p14*p18*p18+p13*p0*p0*p65*p65+p17*p66*p66*p0*p0+p0*p16*p16*p14*p14+p15*p0*p0*p67*p67+p13*p68*p68*p0*p0)/72;
     AVcorr[531]+=(p0*p12*p9+p11*p0*p51+p10*p52*p0+p0*p14*p9+p13*p0*p53+p10*p54*p0+p0*p16*p9+p15*p0*p55+p10*p56*p0+p0*p20*p9+p19*p0*p57+p10*p58*p0+p0*p10*p11+p9*p0*p51+p12*p52*p0+p0*p14*p11+p13*p0*p59+p12*p60*p0+p0*p18*p11+p17*p0*p61+p12*p62*p0+p0*p11*p19+p12*p0*p63+p20*p64*p0+p0*p10*p13+p9*p0*p53+p14*p54*p0+p0*p12*p13+p11*p0*p59+p14*p60*p0+p0*p17*p13+p18*p0*p65+p14*p66*p0+p0*p13*p15+p14*p0*p67+p16*p68*p0+p0*p10*p15+p9*p0*p55+p16*p56*p0+p0*p18*p15+p17*p0*p69+p16*p70*p0+p0*p20*p15+p19*p0*p71+p16*p72*p0+p0*p12*p17+p11*p0*p61+p18*p62*p0+p0*p16*p17+p15*p0*p69+p18*p70*p0+p0*p19*p17+p20*p0*p73+p18*p74*p0+p0*p10*p19+p9*p0*p57+p20*p58*p0+p0*p16*p19+p15*p0*p71+p20*p72*p0+p0*p20*p12+p19*p0*p63+p11*p64*p0+p0*p18*p20+p17*p0*p73+p19*p74*p0+p0*p14*p18+p13*p0*p65+p17*p66*p0+p0*p16*p14+p15*p0*p67+p13*p68*p0)/72;
     AVcorr[540]+=(p0*p0*p9*p25+p10*p10*p0*p60+p26*p26*p59*p0+p0*p0*p9*p26+p10*p10*p0*p72+p25*p25*p71*p0+p0*p0*p11*p24+p12*p12*p0*p54+p23*p23*p53*p0+p0*p0*p11*p23+p12*p12*p0*p74+p24*p24*p73*p0+p0*p0*p13*p22+p14*p14*p0*p52+p21*p21*p51*p0+p0*p0*p13*p21+p14*p14*p0*p69+p22*p22*p70*p0+p0*p0*p15*p24+p16*p16*p0*p58+p23*p23*p57*p0+p0*p0*p15*p23+p16*p16*p0*p65+p24*p24*p66*p0+p0*p0*p17*p25+p18*p18*p0*p63+p26*p26*p64*p0+p0*p0*p17*p26+p18*p18*p0*p68+p25*p25*p67*p0+p0*p0*p19*p22+p20*p20*p0*p56+p21*p21*p55*p0+p0*p0*p19*p21+p20*p20*p0*p61+p22*p22*p62*p0+p0*p0*p20*p22+p19*p19*p0*p62+p21*p21*p61*p0+p0*p0*p20*p21+p19*p19*p0*p55+p22*p22*p56*p0+p0*p0*p18*p25+p17*p17*p0*p67+p26*p26*p68*p0+p0*p0*p18*p26+p17*p17*p0*p64+p25*p25*p63*p0+p0*p0*p16*p24+p15*p15*p0*p66+p23*p23*p65*p0+p0*p0*p16*p23+p15*p15*p0*p57+p24*p24*p58*p0+p0*p0*p14*p22+p13*p13*p0*p70+p21*p21*p69*p0+p0*p0*p14*p21+p13*p13*p0*p51+p22*p22*p52*p0+p0*p0*p12*p24+p11*p11*p0*p73+p23*p23*p74*p0+p0*p0*p12*p23+p11*p11*p0*p53+p24*p24*p54*p0+p0*p0*p10*p25+p9*p9*p0*p71+p26*p26*p72*p0+p0*p0*p10*p26+p9*p9*p0*p59+p25*p25*p60*p0)/72;
     AVcorr[543]+=(p0*p9*p25*p25+p10*p0*p60*p60+p26*p59*p0*p0+p0*p9*p26*p26+p10*p0*p72*p72+p25*p71*p0*p0+p0*p11*p24*p24+p12*p0*p54*p54+p23*p53*p0*p0+p0*p11*p23*p23+p12*p0*p74*p74+p24*p73*p0*p0+p0*p13*p22*p22+p14*p0*p52*p52+p21*p51*p0*p0+p0*p13*p21*p21+p14*p0*p69*p69+p22*p70*p0*p0+p0*p15*p24*p24+p16*p0*p58*p58+p23*p57*p0*p0+p0*p15*p23*p23+p16*p0*p65*p65+p24*p66*p0*p0+p0*p17*p25*p25+p18*p0*p63*p63+p26*p64*p0*p0+p0*p17*p26*p26+p18*p0*p68*p68+p25*p67*p0*p0+p0*p19*p22*p22+p20*p0*p56*p56+p21*p55*p0*p0+p0*p19*p21*p21+p20*p0*p61*p61+p22*p62*p0*p0+p0*p20*p22*p22+p19*p0*p62*p62+p21*p61*p0*p0+p0*p20*p21*p21+p19*p0*p55*p55+p22*p56*p0*p0+p0*p18*p25*p25+p17*p0*p67*p67+p26*p68*p0*p0+p0*p18*p26*p26+p17*p0*p64*p64+p25*p63*p0*p0+p0*p16*p24*p24+p15*p0*p66*p66+p23*p65*p0*p0+p0*p16*p23*p23+p15*p0*p57*p57+p24*p58*p0*p0+p0*p14*p22*p22+p13*p0*p70*p70+p21*p69*p0*p0+p0*p14*p21*p21+p13*p0*p51*p51+p22*p52*p0*p0+p0*p12*p24*p24+p11*p0*p73*p73+p23*p74*p0*p0+p0*p12*p23*p23+p11*p0*p53*p53+p24*p54*p0*p0+p0*p10*p25*p25+p9*p0*p71*p71+p26*p72*p0*p0+p0*p10*p26*p26+p9*p0*p59*p59+p25*p60*p0*p0)/72;
     AVcorr[547]+=(p0*p9*p25+p10*p0*p60+p26*p59*p0+p0*p9*p26+p10*p0*p72+p25*p71*p0+p0*p11*p24+p12*p0*p54+p23*p53*p0+p0*p11*p23+p12*p0*p74+p24*p73*p0+p0*p13*p22+p14*p0*p52+p21*p51*p0+p0*p13*p21+p14*p0*p69+p22*p70*p0+p0*p15*p24+p16*p0*p58+p23*p57*p0+p0*p15*p23+p16*p0*p65+p24*p66*p0+p0*p17*p25+p18*p0*p63+p26*p64*p0+p0*p17*p26+p18*p0*p68+p25*p67*p0+p0*p19*p22+p20*p0*p56+p21*p55*p0+p0*p19*p21+p20*p0*p61+p22*p62*p0+p0*p20*p22+p19*p0*p62+p21*p61*p0+p0*p20*p21+p19*p0*p55+p22*p56*p0+p0*p18*p25+p17*p0*p67+p26*p68*p0+p0*p18*p26+p17*p0*p64+p25*p63*p0+p0*p16*p24+p15*p0*p66+p23*p65*p0+p0*p16*p23+p15*p0*p57+p24*p58*p0+p0*p14*p22+p13*p0*p70+p21*p69*p0+p0*p14*p21+p13*p0*p51+p22*p52*p0+p0*p12*p24+p11*p0*p73+p23*p74*p0+p0*p12*p23+p11*p0*p53+p24*p54*p0+p0*p10*p25+p9*p0*p71+p26*p72*p0+p0*p10*p26+p9*p0*p59+p25*p60*p0)/72;
     AVcorr[557]+=(p103*p0*p0*p70+p45*p69*p69*p0+p99*p0*p0*p73+p47*p74*p74*p0+p115*p0*p0*p62+p39*p61*p61*p0+p107*p0*p0*p66+p43*p65*p65*p0+p95*p0*p0*p71+p49*p72*p72*p0+p116*p0*p0*p56+p37*p55*p55*p0+p111*p0*p0*p67+p41*p68*p68*p0+p108*p0*p0*p58+p33*p57*p57*p0+p112*p0*p0*p63+p35*p64*p64*p0+p104*p0*p0*p52+p31*p51*p51*p0+p97*p0*p0*p59+p50*p60*p60*p0+p113*p0*p0*p64+p42*p63*p63*p0+p101*p0*p0*p53+p48*p54*p54*p0+p109*p0*p0*p57+p44*p58*p58*p0+p100*p0*p0*p54+p29*p53*p53*p0+p114*p0*p0*p68+p36*p67*p67*p0+p96*p0*p0*p60+p27*p59*p59*p0+p110*p0*p0*p65+p34*p66*p66*p0+p105*p0*p0*p51+p46*p52*p52*p0+p117*p0*p0*p55+p40*p56*p56*p0+p118*p0*p0*p61+p38*p62*p62*p0+p98*p0*p0*p72+p28*p71*p71*p0+p102*p0*p0*p74+p30*p73*p73*p0+p106*p0*p0*p69+p32*p70*p70*p0)/72;
     AVcorr[564]+=(p7*p7*p0*p52+p29*p29*p51*p0+p7*p7*p0*p54+p31*p31*p53*p0+p8*p8*p0*p56+p33*p33*p55*p0+p8*p8*p0*p58+p37*p37*p57*p0+p6*p6*p0*p52+p27*p27*p51*p0+p6*p6*p0*p60+p31*p31*p59*p0+p8*p8*p0*p62+p35*p35*p61*p0+p8*p8*p0*p63+p39*p39*p64*p0+p5*p5*p0*p54+p27*p27*p53*p0+p5*p5*p0*p60+p29*p29*p59*p0+p8*p8*p0*p66+p41*p41*p65*p0+p8*p8*p0*p67+p43*p43*p68*p0+p4*p4*p0*p56+p28*p28*p55*p0+p7*p7*p0*p70+p36*p36*p69*p0+p4*p4*p0*p72+p37*p37*p71*p0+p7*p7*p0*p68+p45*p45*p67*p0+p4*p4*p0*p62+p30*p30*p61*p0+p6*p6*p0*p70+p34*p34*p69*p0+p4*p4*p0*p74+p39*p39*p73*p0+p6*p6*p0*p65+p45*p45*p66*p0+p3*p3*p0*p58+p28*p28*p57*p0+p3*p3*p0*p72+p33*p33*p71*p0+p7*p7*p0*p73+p42*p42*p74*p0+p7*p7*p0*p64+p47*p47*p63*p0+p2*p2*p0*p63+p30*p30*p64*p0+p2*p2*p0*p74+p35*p35*p73*p0+p6*p6*p0*p71+p44*p44*p72*p0+p6*p6*p0*p57+p49*p49*p58*p0+p3*p3*p0*p66+p32*p32*p65*p0+p5*p5*p0*p73+p38*p38*p74*p0+p3*p3*p0*p69+p43*p43*p70*p0+p5*p5*p0*p61+p47*p47*p62*p0+p2*p2*p0*p67+p32*p32*p68*p0+p5*p5*p0*p71+p40*p40*p72*p0+p2*p2*p0*p69+p41*p41*p70*p0+p5*p5*p0*p55+p49*p49*p56*p0+p1*p1*p0*p68+p34*p34*p67*p0+p1*p1*p0*p65+p36*p36*p66*p0+p4*p4*p0*p59+p48*p48*p60*p0+p4*p4*p0*p53+p50*p50*p54*p0+p1*p1*p0*p64+p38*p38*p63*p0+p1*p1*p0*p61+p42*p42*p62*p0+p3*p3*p0*p59+p46*p46*p60*p0+p3*p3*p0*p51+p50*p50*p52*p0+p1*p1*p0*p57+p40*p40*p58*p0+p1*p1*p0*p55+p44*p44*p56*p0+p2*p2*p0*p53+p46*p46*p54*p0+p2*p2*p0*p51+p48*p48*p52*p0)/144;
     AVcorr[573]+=(p0*p0*p10*p10*p62+p9*p9*p0*p0*p74+p61*p61*p73*p73*p0+p0*p0*p10*p10*p66+p9*p9*p0*p0*p69+p65*p65*p70*p70*p0+p0*p0*p10*p10*p70+p9*p9*p0*p0*p65+p69*p69*p66*p66*p0+p0*p0*p10*p10*p73+p9*p9*p0*p0*p61+p74*p74*p62*p62*p0+p0*p0*p12*p12*p56+p11*p11*p0*p0*p72+p55*p55*p71*p71*p0+p0*p0*p12*p12*p67+p11*p11*p0*p0*p69+p68*p68*p70*p70*p0+p0*p0*p12*p12*p70+p11*p11*p0*p0*p68+p69*p69*p67*p67*p0+p0*p0*p12*p12*p71+p11*p11*p0*p0*p55+p72*p72*p56*p56*p0+p0*p0*p14*p14*p58+p13*p13*p0*p0*p72+p57*p57*p71*p71*p0+p0*p0*p14*p14*p63+p13*p13*p0*p0*p74+p64*p64*p73*p73*p0+p0*p0*p14*p14*p73+p13*p13*p0*p0*p64+p74*p74*p63*p63*p0+p0*p0*p14*p14*p71+p13*p13*p0*p0*p57+p72*p72*p58*p58*p0+p0*p0*p16*p16*p52+p15*p15*p0*p0*p60+p51*p51*p59*p59*p0+p0*p0*p16*p16*p62+p15*p15*p0*p0*p63+p61*p61*p64*p64*p0+p0*p0*p16*p16*p64+p15*p15*p0*p0*p61+p63*p63*p62*p62*p0+p0*p0*p16*p16*p59+p15*p15*p0*p0*p51+p60*p60*p52*p52*p0+p0*p0*p18*p18*p52+p17*p17*p0*p0*p54+p51*p51*p53*p53*p0+p0*p0*p18*p18*p56+p17*p17*p0*p0*p58+p55*p55*p57*p57*p0+p0*p0*p18*p18*p57+p17*p17*p0*p0*p55+p58*p58*p56*p56*p0+p0*p0*p18*p18*p53+p17*p17*p0*p0*p51+p54*p54*p52*p52*p0+p0*p0*p20*p20*p54+p19*p19*p0*p0*p60+p53*p53*p59*p59*p0+p0*p0*p20*p20*p68+p19*p19*p0*p0*p65+p67*p67*p66*p66*p0+p0*p0*p20*p20*p66+p19*p19*p0*p0*p67+p65*p65*p68*p68*p0+p0*p0*p20*p20*p59+p19*p19*p0*p0*p53+p60*p60*p54*p54*p0)/72;
     AVcorr[574]+=(p0*p10*p62*p62+p9*p0*p74*p74+p61*p73*p0*p0+p0*p10*p66*p66+p9*p0*p69*p69+p65*p70*p0*p0+p0*p10*p70*p70+p9*p0*p65*p65+p69*p66*p0*p0+p0*p10*p73*p73+p9*p0*p61*p61+p74*p62*p0*p0+p0*p12*p56*p56+p11*p0*p72*p72+p55*p71*p0*p0+p0*p12*p67*p67+p11*p0*p69*p69+p68*p70*p0*p0+p0*p12*p70*p70+p11*p0*p68*p68+p69*p67*p0*p0+p0*p12*p71*p71+p11*p0*p55*p55+p72*p56*p0*p0+p0*p14*p58*p58+p13*p0*p72*p72+p57*p71*p0*p0+p0*p14*p63*p63+p13*p0*p74*p74+p64*p73*p0*p0+p0*p14*p73*p73+p13*p0*p64*p64+p74*p63*p0*p0+p0*p14*p71*p71+p13*p0*p57*p57+p72*p58*p0*p0+p0*p16*p52*p52+p15*p0*p60*p60+p51*p59*p0*p0+p0*p16*p62*p62+p15*p0*p63*p63+p61*p64*p0*p0+p0*p16*p64*p64+p15*p0*p61*p61+p63*p62*p0*p0+p0*p16*p59*p59+p15*p0*p51*p51+p60*p52*p0*p0+p0*p18*p52*p52+p17*p0*p54*p54+p51*p53*p0*p0+p0*p18*p56*p56+p17*p0*p58*p58+p55*p57*p0*p0+p0*p18*p57*p57+p17*p0*p55*p55+p58*p56*p0*p0+p0*p18*p53*p53+p17*p0*p51*p51+p54*p52*p0*p0+p0*p20*p54*p54+p19*p0*p60*p60+p53*p59*p0*p0+p0*p20*p68*p68+p19*p0*p65*p65+p67*p66*p0*p0+p0*p20*p66*p66+p19*p0*p67*p67+p65*p68*p0*p0+p0*p20*p59*p59+p19*p0*p53*p53+p60*p54*p0*p0)/72;
     AVcorr[576]+=(p0*p0*p10*p10*p62*p62+p9*p9*p0*p0*p74*p74+p61*p61*p73*p73*p0*p0+p0*p0*p10*p10*p66*p66+p9*p9*p0*p0*p69*p69+p65*p65*p70*p70*p0*p0+p0*p0*p10*p10*p70*p70+p9*p9*p0*p0*p65*p65+p69*p69*p66*p66*p0*p0+p0*p0*p10*p10*p73*p73+p9*p9*p0*p0*p61*p61+p74*p74*p62*p62*p0*p0+p0*p0*p12*p12*p56*p56+p11*p11*p0*p0*p72*p72+p55*p55*p71*p71*p0*p0+p0*p0*p12*p12*p67*p67+p11*p11*p0*p0*p69*p69+p68*p68*p70*p70*p0*p0+p0*p0*p12*p12*p70*p70+p11*p11*p0*p0*p68*p68+p69*p69*p67*p67*p0*p0+p0*p0*p12*p12*p71*p71+p11*p11*p0*p0*p55*p55+p72*p72*p56*p56*p0*p0+p0*p0*p14*p14*p58*p58+p13*p13*p0*p0*p72*p72+p57*p57*p71*p71*p0*p0+p0*p0*p14*p14*p63*p63+p13*p13*p0*p0*p74*p74+p64*p64*p73*p73*p0*p0+p0*p0*p14*p14*p73*p73+p13*p13*p0*p0*p64*p64+p74*p74*p63*p63*p0*p0+p0*p0*p14*p14*p71*p71+p13*p13*p0*p0*p57*p57+p72*p72*p58*p58*p0*p0+p0*p0*p16*p16*p52*p52+p15*p15*p0*p0*p60*p60+p51*p51*p59*p59*p0*p0+p0*p0*p16*p16*p62*p62+p15*p15*p0*p0*p63*p63+p61*p61*p64*p64*p0*p0+p0*p0*p16*p16*p64*p64+p15*p15*p0*p0*p61*p61+p63*p63*p62*p62*p0*p0+p0*p0*p16*p16*p59*p59+p15*p15*p0*p0*p51*p51+p60*p60*p52*p52*p0*p0+p0*p0*p18*p18*p52*p52+p17*p17*p0*p0*p54*p54+p51*p51*p53*p53*p0*p0+p0*p0*p18*p18*p56*p56+p17*p17*p0*p0*p58*p58+p55*p55*p57*p57*p0*p0+p0*p0*p18*p18*p57*p57+p17*p17*p0*p0*p55*p55+p58*p58*p56*p56*p0*p0+p0*p0*p18*p18*p53*p53+p17*p17*p0*p0*p51*p51+p54*p54*p52*p52*p0*p0+p0*p0*p20*p20*p54*p54+p19*p19*p0*p0*p60*p60+p53*p53*p59*p59*p0*p0+p0*p0*p20*p20*p68*p68+p19*p19*p0*p0*p65*p65+p67*p67*p66*p66*p0*p0+p0*p0*p20*p20*p66*p66+p19*p19*p0*p0*p67*p67+p65*p65*p68*p68*p0*p0+p0*p0*p20*p20*p59*p59+p19*p19*p0*p0*p53*p53+p60*p60*p54*p54*p0*p0)/72;
     AVcorr[577]+=(p0*p10*p62+p9*p0*p74+p61*p73*p0+p0*p10*p66+p9*p0*p69+p65*p70*p0+p0*p10*p70+p9*p0*p65+p69*p66*p0+p0*p10*p73+p9*p0*p61+p74*p62*p0+p0*p12*p56+p11*p0*p72+p55*p71*p0+p0*p12*p67+p11*p0*p69+p68*p70*p0+p0*p12*p70+p11*p0*p68+p69*p67*p0+p0*p12*p71+p11*p0*p55+p72*p56*p0+p0*p14*p58+p13*p0*p72+p57*p71*p0+p0*p14*p63+p13*p0*p74+p64*p73*p0+p0*p14*p73+p13*p0*p64+p74*p63*p0+p0*p14*p71+p13*p0*p57+p72*p58*p0+p0*p16*p52+p15*p0*p60+p51*p59*p0+p0*p16*p62+p15*p0*p63+p61*p64*p0+p0*p16*p64+p15*p0*p61+p63*p62*p0+p0*p16*p59+p15*p0*p51+p60*p52*p0+p0*p18*p52+p17*p0*p54+p51*p53*p0+p0*p18*p56+p17*p0*p58+p55*p57*p0+p0*p18*p57+p17*p0*p55+p58*p56*p0+p0*p18*p53+p17*p0*p51+p54*p52*p0+p0*p20*p54+p19*p0*p60+p53*p59*p0+p0*p20*p68+p19*p0*p65+p67*p66*p0+p0*p20*p66+p19*p0*p67+p65*p68*p0+p0*p20*p59+p19*p0*p53+p60*p54*p0)/72;
     AVcorr[586]+=(p0*p22*p54*p54+p21*p0*p66*p66+p53*p65*p0*p0+p0*p24*p52*p52+p23*p0*p62*p62+p51*p61*p0*p0+p0*p22*p58*p58+p21*p0*p73*p73+p57*p74*p0*p0+p0*p24*p56*p56+p23*p0*p70*p70+p55*p69*p0*p0+p0*p22*p60*p60+p21*p0*p67*p67+p59*p68*p0*p0+p0*p26*p56*p56+p25*p0*p52*p52+p55*p51*p0*p0+p0*p22*p63*p63+p21*p0*p71*p71+p64*p72*p0*p0+p0*p26*p70*p70+p25*p0*p62*p62+p69*p61*p0*p0+p0*p24*p60*p60+p23*p0*p63*p63+p59*p64*p0*p0+p0*p26*p58*p58+p25*p0*p54*p54+p57*p53*p0*p0+p0*p24*p67*p67+p23*p0*p71*p71+p68*p72*p0*p0+p0*p26*p73*p73+p25*p0*p66*p66+p74*p65*p0*p0+p0*p22*p72*p72+p21*p0*p64*p64+p71*p63*p0*p0+p0*p22*p68*p68+p21*p0*p59*p59+p67*p60*p0*p0+p0*p22*p74*p74+p21*p0*p57*p57+p73*p58*p0*p0+p0*p22*p65*p65+p21*p0*p53*p53+p66*p54*p0*p0+p0*p24*p72*p72+p23*p0*p68*p68+p71*p67*p0*p0+p0*p24*p64*p64+p23*p0*p59*p59+p63*p60*p0*p0+p0*p26*p65*p65+p25*p0*p74*p74+p66*p73*p0*p0+p0*p26*p53*p53+p25*p0*p57*p57+p54*p58*p0*p0+p0*p24*p69*p69+p23*p0*p55*p55+p70*p56*p0*p0+p0*p24*p61*p61+p23*p0*p51*p51+p62*p52*p0*p0+p0*p26*p61*p61+p25*p0*p69*p69+p62*p70*p0*p0+p0*p26*p51*p51+p25*p0*p55*p55+p52*p56*p0*p0)/72;
     AVcorr[587]+=(p0*p0*p22*p54*p54+p21*p21*p0*p66*p66+p53*p53*p65*p0*p0+p0*p0*p21*p66*p66+p22*p22*p0*p54*p54+p65*p65*p53*p0*p0+p0*p0*p24*p52*p52+p23*p23*p0*p62*p62+p51*p51*p61*p0*p0+p0*p0*p23*p62*p62+p24*p24*p0*p52*p52+p61*p61*p51*p0*p0+p0*p0*p22*p58*p58+p21*p21*p0*p73*p73+p57*p57*p74*p0*p0+p0*p0*p21*p73*p73+p22*p22*p0*p58*p58+p74*p74*p57*p0*p0+p0*p0*p24*p56*p56+p23*p23*p0*p70*p70+p55*p55*p69*p0*p0+p0*p0*p23*p70*p70+p24*p24*p0*p56*p56+p69*p69*p55*p0*p0+p0*p0*p22*p60*p60+p21*p21*p0*p67*p67+p59*p59*p68*p0*p0+p0*p0*p21*p67*p67+p22*p22*p0*p60*p60+p68*p68*p59*p0*p0+p0*p0*p26*p56*p56+p25*p25*p0*p52*p52+p55*p55*p51*p0*p0+p0*p0*p25*p52*p52+p26*p26*p0*p56*p56+p51*p51*p55*p0*p0+p0*p0*p22*p63*p63+p21*p21*p0*p71*p71+p64*p64*p72*p0*p0+p0*p0*p21*p71*p71+p22*p22*p0*p63*p63+p72*p72*p64*p0*p0+p0*p0*p26*p70*p70+p25*p25*p0*p62*p62+p69*p69*p61*p0*p0+p0*p0*p25*p62*p62+p26*p26*p0*p70*p70+p61*p61*p69*p0*p0+p0*p0*p24*p60*p60+p23*p23*p0*p63*p63+p59*p59*p64*p0*p0+p0*p0*p23*p63*p63+p24*p24*p0*p60*p60+p64*p64*p59*p0*p0+p0*p0*p26*p58*p58+p25*p25*p0*p54*p54+p57*p57*p53*p0*p0+p0*p0*p25*p54*p54+p26*p26*p0*p58*p58+p53*p53*p57*p0*p0+p0*p0*p24*p67*p67+p23*p23*p0*p71*p71+p68*p68*p72*p0*p0+p0*p0*p23*p71*p71+p24*p24*p0*p67*p67+p72*p72*p68*p0*p0+p0*p0*p26*p73*p73+p25*p25*p0*p66*p66+p74*p74*p65*p0*p0+p0*p0*p25*p66*p66+p26*p26*p0*p73*p73+p65*p65*p74*p0*p0+p0*p0*p22*p72*p72+p21*p21*p0*p64*p64+p71*p71*p63*p0*p0+p0*p0*p21*p64*p64+p22*p22*p0*p72*p72+p63*p63*p71*p0*p0+p0*p0*p22*p68*p68+p21*p21*p0*p59*p59+p67*p67*p60*p0*p0+p0*p0*p21*p59*p59+p22*p22*p0*p68*p68+p60*p60*p67*p0*p0+p0*p0*p22*p74*p74+p21*p21*p0*p57*p57+p73*p73*p58*p0*p0+p0*p0*p21*p57*p57+p22*p22*p0*p74*p74+p58*p58*p73*p0*p0+p0*p0*p22*p65*p65+p21*p21*p0*p53*p53+p66*p66*p54*p0*p0+p0*p0*p21*p53*p53+p22*p22*p0*p65*p65+p54*p54*p66*p0*p0+p0*p0*p24*p72*p72+p23*p23*p0*p68*p68+p71*p71*p67*p0*p0+p0*p0*p23*p68*p68+p24*p24*p0*p72*p72+p67*p67*p71*p0*p0+p0*p0*p24*p64*p64+p23*p23*p0*p59*p59+p63*p63*p60*p0*p0+p0*p0*p23*p59*p59+p24*p24*p0*p64*p64+p60*p60*p63*p0*p0+p0*p0*p26*p65*p65+p25*p25*p0*p74*p74+p66*p66*p73*p0*p0+p0*p0*p25*p74*p74+p26*p26*p0*p65*p65+p73*p73*p66*p0*p0+p0*p0*p26*p53*p53+p25*p25*p0*p57*p57+p54*p54*p58*p0*p0+p0*p0*p25*p57*p57+p26*p26*p0*p53*p53+p58*p58*p54*p0*p0+p0*p0*p24*p69*p69+p23*p23*p0*p55*p55+p70*p70*p56*p0*p0+p0*p0*p23*p55*p55+p24*p24*p0*p69*p69+p56*p56*p70*p0*p0+p0*p0*p24*p61*p61+p23*p23*p0*p51*p51+p62*p62*p52*p0*p0+p0*p0*p23*p51*p51+p24*p24*p0*p61*p61+p52*p52*p62*p0*p0+p0*p0*p26*p61*p61+p25*p25*p0*p69*p69+p62*p62*p70*p0*p0+p0*p0*p25*p69*p69+p26*p26*p0*p61*p61+p70*p70*p62*p0*p0+p0*p0*p26*p51*p51+p25*p25*p0*p55*p55+p52*p52*p56*p0*p0+p0*p0*p25*p55*p55+p26*p26*p0*p51*p51+p56*p56*p52*p0*p0)/144;
     AVcorr[589]+=(p0*p22*p54+p21*p0*p66+p53*p65*p0+p0*p24*p52+p23*p0*p62+p51*p61*p0+p0*p22*p58+p21*p0*p73+p57*p74*p0+p0*p24*p56+p23*p0*p70+p55*p69*p0+p0*p22*p60+p21*p0*p67+p59*p68*p0+p0*p26*p56+p25*p0*p52+p55*p51*p0+p0*p22*p63+p21*p0*p71+p64*p72*p0+p0*p26*p70+p25*p0*p62+p69*p61*p0+p0*p24*p60+p23*p0*p63+p59*p64*p0+p0*p26*p58+p25*p0*p54+p57*p53*p0+p0*p24*p67+p23*p0*p71+p68*p72*p0+p0*p26*p73+p25*p0*p66+p74*p65*p0+p0*p22*p72+p21*p0*p64+p71*p63*p0+p0*p22*p68+p21*p0*p59+p67*p60*p0+p0*p22*p74+p21*p0*p57+p73*p58*p0+p0*p22*p65+p21*p0*p53+p66*p54*p0+p0*p24*p72+p23*p0*p68+p71*p67*p0+p0*p24*p64+p23*p0*p59+p63*p60*p0+p0*p26*p65+p25*p0*p74+p66*p73*p0+p0*p26*p53+p25*p0*p57+p54*p58*p0+p0*p24*p69+p23*p0*p55+p70*p56*p0+p0*p24*p61+p23*p0*p51+p62*p52*p0+p0*p26*p61+p25*p0*p69+p62*p70*p0+p0*p26*p51+p25*p0*p55+p52*p56*p0)/72;
     AVcorr[634]+=(p0*p0*p54*p56+p53*p53*p0*p68+p55*p55*p67*p0+p0*p0*p67*p55+p68*p68*p0*p53+p56*p56*p54*p0+p0*p0*p68*p53+p67*p67*p0*p55+p54*p54*p56*p0+p0*p0*p52*p58+p51*p51*p0*p64+p57*p57*p63*p0+p0*p0*p63*p57+p64*p64*p0*p51+p58*p58*p52*p0+p0*p0*p64*p51+p63*p63*p0*p57+p52*p52*p58*p0+p0*p0*p66*p61+p65*p65*p0*p59+p62*p62*p60*p0+p0*p0*p62*p60+p61*p61*p0*p66+p59*p59*p65*p0+p0*p0*p59*p65+p60*p60*p0*p62+p66*p66*p61*p0+p0*p0*p57*p51+p58*p58*p0*p64+p52*p52*p63*p0+p0*p0*p58*p64+p57*p57*p0*p51+p63*p63*p52*p0+p0*p0*p52*p63+p51*p51*p0*p57+p64*p64*p58*p0+p0*p0*p62*p65+p61*p61*p0*p59+p66*p66*p60*p0+p0*p0*p59*p61+p60*p60*p0*p66+p62*p62*p65*p0+p0*p0*p66*p60+p65*p65*p0*p62+p59*p59*p61*p0+p0*p0*p55*p53+p56*p56*p0*p68+p54*p54*p67*p0+p0*p0*p56*p68+p55*p55*p0*p53+p67*p67*p54*p0+p0*p0*p54*p67+p53*p53*p0*p55+p68*p68*p56*p0+p0*p0*p73*p69+p74*p74*p0*p71+p70*p70*p72*p0+p0*p0*p70*p72+p69*p69*p0*p73+p71*p71*p74*p0+p0*p0*p71*p74+p72*p72*p0*p70+p73*p73*p69*p0+p0*p0*p73*p72+p74*p74*p0*p70+p71*p71*p69*p0+p0*p0*p70*p74+p69*p69*p0*p71+p73*p73*p72*p0+p0*p0*p71*p69+p72*p72*p0*p73+p70*p70*p74*p0)/72;
     AVcorr[636]+=(p0*p0*p54*p54*p56*p56+p53*p53*p0*p0*p68*p68+p55*p55*p67*p67*p0*p0+p0*p0*p52*p52*p58*p58+p51*p51*p0*p0*p64*p64+p57*p57*p63*p63*p0*p0+p0*p0*p66*p66*p61*p61+p65*p65*p0*p0*p59*p59+p62*p62*p60*p60*p0*p0+p0*p0*p57*p57*p51*p51+p58*p58*p0*p0*p64*p64+p52*p52*p63*p63*p0*p0+p0*p0*p62*p62*p65*p65+p61*p61*p0*p0*p59*p59+p66*p66*p60*p60*p0*p0+p0*p0*p55*p55*p53*p53+p56*p56*p0*p0*p68*p68+p54*p54*p67*p67*p0*p0+p0*p0*p73*p73*p69*p69+p74*p74*p0*p0*p71*p71+p70*p70*p72*p72*p0*p0+p0*p0*p73*p73*p72*p72+p74*p74*p0*p0*p70*p70+p71*p71*p69*p69*p0*p0)/24;
     AVcorr[654]+=(p103*p103*p0*p119+p99*p99*p0*p120+p115*p115*p0*p121+p107*p107*p0*p122+p95*p95*p0*p123+p116*p116*p0*p124+p111*p111*p0*p125+p108*p108*p0*p126+p112*p112*p0*p127+p104*p104*p0*p128+p97*p97*p0*p129+p113*p113*p0*p130+p101*p101*p0*p131+p109*p109*p0*p132+p100*p100*p0*p133+p114*p114*p0*p134+p96*p96*p0*p135+p110*p110*p0*p136+p105*p105*p0*p137+p117*p117*p0*p138+p118*p118*p0*p139+p98*p98*p0*p140+p102*p102*p0*p141+p106*p106*p0*p142)/72;
     AVcorr[655]+=(p103*p0*p0*p119+p99*p0*p0*p120+p115*p0*p0*p121+p107*p0*p0*p122+p95*p0*p0*p123+p116*p0*p0*p124+p111*p0*p0*p125+p108*p0*p0*p126+p112*p0*p0*p127+p104*p0*p0*p128+p97*p0*p0*p129+p113*p0*p0*p130+p101*p0*p0*p131+p109*p0*p0*p132+p100*p0*p0*p133+p114*p0*p0*p134+p96*p0*p0*p135+p110*p0*p0*p136+p105*p0*p0*p137+p117*p0*p0*p138+p118*p0*p0*p139+p98*p0*p0*p140+p102*p0*p0*p141+p106*p0*p0*p142)/72;
     AVcorr[657]+=(p103*p0*p119*p119+p99*p0*p120*p120+p115*p0*p121*p121+p107*p0*p122*p122+p95*p0*p123*p123+p116*p0*p124*p124+p111*p0*p125*p125+p108*p0*p126*p126+p112*p0*p127*p127+p104*p0*p128*p128+p97*p0*p129*p129+p113*p0*p130*p130+p101*p0*p131*p131+p109*p0*p132*p132+p100*p0*p133*p133+p114*p0*p134*p134+p96*p0*p135*p135+p110*p0*p136*p136+p105*p0*p137*p137+p117*p0*p138*p138+p118*p0*p139*p139+p98*p0*p140*p140+p102*p0*p141*p141+p106*p0*p142*p142)/72;
     AVcorr[762]+=(p0*p0*p9*p42+p10*p10*p0*p121+p0*p0*p9*p36+p10*p10*p0*p122+p0*p0*p9*p41+p10*p10*p0*p119+p0*p0*p9*p35+p10*p10*p0*p120+p0*p0*p11*p44+p12*p12*p0*p124+p0*p0*p11*p34+p12*p12*p0*p125+p0*p0*p11*p43+p12*p12*p0*p119+p0*p0*p11*p33+p12*p12*p0*p123+p0*p0*p13*p40+p14*p14*p0*p126+p0*p0*p13*p38+p14*p14*p0*p127+p0*p0*p13*p39+p14*p14*p0*p120+p0*p0*p13*p37+p14*p14*p0*p123+p0*p0*p15*p48+p16*p16*p0*p128+p0*p0*p15*p47+p16*p16*p0*p121+p0*p0*p15*p30+p16*p16*p0*p130+p0*p0*p15*p29+p16*p16*p0*p129+p0*p0*p17*p50+p18*p18*p0*p128+p0*p0*p17*p49+p18*p18*p0*p124+p0*p0*p17*p28+p18*p18*p0*p132+p0*p0*p17*p27+p18*p18*p0*p131+p0*p0*p19*p46+p20*p20*p0*p133+p0*p0*p19*p32+p20*p20*p0*p134+p0*p0*p19*p45+p20*p20*p0*p122+p0*p0*p19*p31+p20*p20*p0*p129+p0*p0*p20*p46+p19*p19*p0*p135+p0*p0*p20*p32+p19*p19*p0*p136+p0*p0*p20*p45+p19*p19*p0*p125+p0*p0*p20*p31+p19*p19*p0*p131+p0*p0*p18*p50+p17*p17*p0*p133+p0*p0*p18*p49+p17*p17*p0*p126+p0*p0*p18*p28+p17*p17*p0*p138+p0*p0*p18*p27+p17*p17*p0*p137+p0*p0*p16*p48+p15*p15*p0*p135+p0*p0*p16*p47+p15*p15*p0*p127+p0*p0*p16*p30+p15*p15*p0*p139+p0*p0*p16*p29+p15*p15*p0*p137+p0*p0*p14*p40+p13*p13*p0*p140+p0*p0*p14*p38+p13*p13*p0*p141+p0*p0*p14*p39+p13*p13*p0*p130+p0*p0*p14*p37+p13*p13*p0*p132+p0*p0*p12*p44+p11*p11*p0*p140+p0*p0*p12*p34+p11*p11*p0*p142+p0*p0*p12*p43+p11*p11*p0*p134+p0*p0*p12*p33+p11*p11*p0*p138+p0*p0*p10*p42+p9*p9*p0*p141+p0*p0*p10*p36+p9*p9*p0*p142+p0*p0*p10*p41+p9*p9*p0*p136+p0*p0*p10*p35+p9*p9*p0*p139)/144;
     AVcorr[763]+=(p0*p9*p9*p42+p10*p0*p0*p121+p0*p9*p9*p36+p10*p0*p0*p122+p0*p9*p9*p41+p10*p0*p0*p119+p0*p9*p9*p35+p10*p0*p0*p120+p0*p11*p11*p44+p12*p0*p0*p124+p0*p11*p11*p34+p12*p0*p0*p125+p0*p11*p11*p43+p12*p0*p0*p119+p0*p11*p11*p33+p12*p0*p0*p123+p0*p13*p13*p40+p14*p0*p0*p126+p0*p13*p13*p38+p14*p0*p0*p127+p0*p13*p13*p39+p14*p0*p0*p120+p0*p13*p13*p37+p14*p0*p0*p123+p0*p15*p15*p48+p16*p0*p0*p128+p0*p15*p15*p47+p16*p0*p0*p121+p0*p15*p15*p30+p16*p0*p0*p130+p0*p15*p15*p29+p16*p0*p0*p129+p0*p17*p17*p50+p18*p0*p0*p128+p0*p17*p17*p49+p18*p0*p0*p124+p0*p17*p17*p28+p18*p0*p0*p132+p0*p17*p17*p27+p18*p0*p0*p131+p0*p19*p19*p46+p20*p0*p0*p133+p0*p19*p19*p32+p20*p0*p0*p134+p0*p19*p19*p45+p20*p0*p0*p122+p0*p19*p19*p31+p20*p0*p0*p129+p0*p20*p20*p46+p19*p0*p0*p135+p0*p20*p20*p32+p19*p0*p0*p136+p0*p20*p20*p45+p19*p0*p0*p125+p0*p20*p20*p31+p19*p0*p0*p131+p0*p18*p18*p50+p17*p0*p0*p133+p0*p18*p18*p49+p17*p0*p0*p126+p0*p18*p18*p28+p17*p0*p0*p138+p0*p18*p18*p27+p17*p0*p0*p137+p0*p16*p16*p48+p15*p0*p0*p135+p0*p16*p16*p47+p15*p0*p0*p127+p0*p16*p16*p30+p15*p0*p0*p139+p0*p16*p16*p29+p15*p0*p0*p137+p0*p14*p14*p40+p13*p0*p0*p140+p0*p14*p14*p38+p13*p0*p0*p141+p0*p14*p14*p39+p13*p0*p0*p130+p0*p14*p14*p37+p13*p0*p0*p132+p0*p12*p12*p44+p11*p0*p0*p140+p0*p12*p12*p34+p11*p0*p0*p142+p0*p12*p12*p43+p11*p0*p0*p134+p0*p12*p12*p33+p11*p0*p0*p138+p0*p10*p10*p42+p9*p0*p0*p141+p0*p10*p10*p36+p9*p0*p0*p142+p0*p10*p10*p41+p9*p0*p0*p136+p0*p10*p10*p35+p9*p0*p0*p139)/144;
     AVcorr[912]+=(p0*p0*p24*p21+p23*p23*p0*p83+p22*p22*p84*p0+p0*p0*p22*p23+p21*p21*p0*p83+p24*p24*p84*p0+p0*p0*p25*p21+p26*p26*p0*p85+p22*p22*p86*p0+p0*p0*p22*p26+p21*p21*p0*p85+p25*p25*p86*p0+p0*p0*p25*p23+p26*p26*p0*p87+p24*p24*p88*p0+p0*p0*p24*p26+p23*p23*p0*p87+p25*p25*p88*p0+p0*p0*p26*p21+p25*p25*p0*p89+p22*p22*p90*p0+p0*p0*p22*p25+p21*p21*p0*p89+p26*p26*p90*p0+p0*p0*p23*p21+p24*p24*p0*p91+p22*p22*p92*p0+p0*p0*p22*p24+p21*p21*p0*p91+p23*p23*p92*p0+p0*p0*p26*p23+p25*p25*p0*p93+p24*p24*p94*p0+p0*p0*p24*p25+p23*p23*p0*p93+p26*p26*p94*p0)/36;
     AVcorr[913]+=(p0*p24*p24*p21+p23*p0*p0*p83+p22*p84*p84*p0+p0*p21*p21*p24+p22*p0*p0*p84+p23*p83*p83*p0+p0*p22*p22*p23+p21*p0*p0*p83+p24*p84*p84*p0+p0*p23*p23*p22+p24*p0*p0*p84+p21*p83*p83*p0+p0*p25*p25*p21+p26*p0*p0*p85+p22*p86*p86*p0+p0*p21*p21*p25+p22*p0*p0*p86+p26*p85*p85*p0+p0*p22*p22*p26+p21*p0*p0*p85+p25*p86*p86*p0+p0*p26*p26*p22+p25*p0*p0*p86+p21*p85*p85*p0+p0*p25*p25*p23+p26*p0*p0*p87+p24*p88*p88*p0+p0*p23*p23*p25+p24*p0*p0*p88+p26*p87*p87*p0+p0*p24*p24*p26+p23*p0*p0*p87+p25*p88*p88*p0+p0*p26*p26*p24+p25*p0*p0*p88+p23*p87*p87*p0+p0*p26*p26*p21+p25*p0*p0*p89+p22*p90*p90*p0+p0*p21*p21*p26+p22*p0*p0*p90+p25*p89*p89*p0+p0*p22*p22*p25+p21*p0*p0*p89+p26*p90*p90*p0+p0*p25*p25*p22+p26*p0*p0*p90+p21*p89*p89*p0+p0*p23*p23*p21+p24*p0*p0*p91+p22*p92*p92*p0+p0*p21*p21*p23+p22*p0*p0*p92+p24*p91*p91*p0+p0*p22*p22*p24+p21*p0*p0*p91+p23*p92*p92*p0+p0*p24*p24*p22+p23*p0*p0*p92+p21*p91*p91*p0+p0*p26*p26*p23+p25*p0*p0*p93+p24*p94*p94*p0+p0*p23*p23*p26+p24*p0*p0*p94+p25*p93*p93*p0+p0*p24*p24*p25+p23*p0*p0*p93+p26*p94*p94*p0+p0*p25*p25*p24+p26*p0*p0*p94+p23*p93*p93*p0)/72;
     AVcorr[914]+=(p0*p0*p24*p24*p21+p23*p23*p0*p0*p83+p22*p22*p84*p84*p0+p0*p0*p21*p21*p24+p22*p22*p0*p0*p84+p23*p23*p83*p83*p0+p0*p0*p22*p22*p23+p21*p21*p0*p0*p83+p24*p24*p84*p84*p0+p0*p0*p23*p23*p22+p24*p24*p0*p0*p84+p21*p21*p83*p83*p0+p0*p0*p25*p25*p21+p26*p26*p0*p0*p85+p22*p22*p86*p86*p0+p0*p0*p21*p21*p25+p22*p22*p0*p0*p86+p26*p26*p85*p85*p0+p0*p0*p22*p22*p26+p21*p21*p0*p0*p85+p25*p25*p86*p86*p0+p0*p0*p26*p26*p22+p25*p25*p0*p0*p86+p21*p21*p85*p85*p0+p0*p0*p25*p25*p23+p26*p26*p0*p0*p87+p24*p24*p88*p88*p0+p0*p0*p23*p23*p25+p24*p24*p0*p0*p88+p26*p26*p87*p87*p0+p0*p0*p24*p24*p26+p23*p23*p0*p0*p87+p25*p25*p88*p88*p0+p0*p0*p26*p26*p24+p25*p25*p0*p0*p88+p23*p23*p87*p87*p0+p0*p0*p26*p26*p21+p25*p25*p0*p0*p89+p22*p22*p90*p90*p0+p0*p0*p21*p21*p26+p22*p22*p0*p0*p90+p25*p25*p89*p89*p0+p0*p0*p22*p22*p25+p21*p21*p0*p0*p89+p26*p26*p90*p90*p0+p0*p0*p25*p25*p22+p26*p26*p0*p0*p90+p21*p21*p89*p89*p0+p0*p0*p23*p23*p21+p24*p24*p0*p0*p91+p22*p22*p92*p92*p0+p0*p0*p21*p21*p23+p22*p22*p0*p0*p92+p24*p24*p91*p91*p0+p0*p0*p22*p22*p24+p21*p21*p0*p0*p91+p23*p23*p92*p92*p0+p0*p0*p24*p24*p22+p23*p23*p0*p0*p92+p21*p21*p91*p91*p0+p0*p0*p26*p26*p23+p25*p25*p0*p0*p93+p24*p24*p94*p94*p0+p0*p0*p23*p23*p26+p24*p24*p0*p0*p94+p25*p25*p93*p93*p0+p0*p0*p24*p24*p25+p23*p23*p0*p0*p93+p26*p26*p94*p94*p0+p0*p0*p25*p25*p24+p26*p26*p0*p0*p94+p23*p23*p93*p93*p0)/72;
     AVcorr[916]+=(p0*p0*p24*p24*p21*p21+p23*p23*p0*p0*p83*p83+p22*p22*p84*p84*p0*p0+p0*p0*p22*p22*p23*p23+p21*p21*p0*p0*p83*p83+p24*p24*p84*p84*p0*p0+p0*p0*p25*p25*p21*p21+p26*p26*p0*p0*p85*p85+p22*p22*p86*p86*p0*p0+p0*p0*p22*p22*p26*p26+p21*p21*p0*p0*p85*p85+p25*p25*p86*p86*p0*p0+p0*p0*p25*p25*p23*p23+p26*p26*p0*p0*p87*p87+p24*p24*p88*p88*p0*p0+p0*p0*p24*p24*p26*p26+p23*p23*p0*p0*p87*p87+p25*p25*p88*p88*p0*p0+p0*p0*p26*p26*p21*p21+p25*p25*p0*p0*p89*p89+p22*p22*p90*p90*p0*p0+p0*p0*p22*p22*p25*p25+p21*p21*p0*p0*p89*p89+p26*p26*p90*p90*p0*p0+p0*p0*p23*p23*p21*p21+p24*p24*p0*p0*p91*p91+p22*p22*p92*p92*p0*p0+p0*p0*p22*p22*p24*p24+p21*p21*p0*p0*p91*p91+p23*p23*p92*p92*p0*p0+p0*p0*p26*p26*p23*p23+p25*p25*p0*p0*p93*p93+p24*p24*p94*p94*p0*p0+p0*p0*p24*p24*p25*p25+p23*p23*p0*p0*p93*p93+p26*p26*p94*p94*p0*p0)/36;
     AVcorr[917]+=(p0*p24*p21+p23*p0*p83+p22*p84*p0+p0*p22*p23+p21*p0*p83+p24*p84*p0+p0*p25*p21+p26*p0*p85+p22*p86*p0+p0*p22*p26+p21*p0*p85+p25*p86*p0+p0*p25*p23+p26*p0*p87+p24*p88*p0+p0*p24*p26+p23*p0*p87+p25*p88*p0+p0*p26*p21+p25*p0*p89+p22*p90*p0+p0*p22*p25+p21*p0*p89+p26*p90*p0+p0*p23*p21+p24*p0*p91+p22*p92*p0+p0*p22*p24+p21*p0*p91+p23*p92*p0+p0*p26*p23+p25*p0*p93+p24*p94*p0+p0*p24*p25+p23*p0*p93+p26*p94*p0)/36;
     AVcorr[986]+=(p0*p0*p10*p10*p9+p9*p9*p0*p0*p83+p10*p10*p84*p84*p0+p0*p0*p9*p9*p10+p10*p10*p0*p0*p84+p9*p9*p83*p83*p0+p0*p0*p12*p12*p11+p11*p11*p0*p0*p85+p12*p12*p86*p86*p0+p0*p0*p11*p11*p12+p12*p12*p0*p0*p86+p11*p11*p85*p85*p0+p0*p0*p14*p14*p13+p13*p13*p0*p0*p87+p14*p14*p88*p88*p0+p0*p0*p13*p13*p14+p14*p14*p0*p0*p88+p13*p13*p87*p87*p0+p0*p0*p16*p16*p15+p15*p15*p0*p0*p89+p16*p16*p90*p90*p0+p0*p0*p15*p15*p16+p16*p16*p0*p0*p90+p15*p15*p89*p89*p0+p0*p0*p18*p18*p17+p17*p17*p0*p0*p91+p18*p18*p92*p92*p0+p0*p0*p17*p17*p18+p18*p18*p0*p0*p92+p17*p17*p91*p91*p0+p0*p0*p20*p20*p19+p19*p19*p0*p0*p93+p20*p20*p94*p94*p0+p0*p0*p19*p19*p20+p20*p20*p0*p0*p94+p19*p19*p93*p93*p0)/36;
     AVcorr[987]+=(p0*p10*p10*p9*p9+p9*p0*p0*p83*p83+p10*p84*p84*p0*p0+p0*p12*p12*p11*p11+p11*p0*p0*p85*p85+p12*p86*p86*p0*p0+p0*p14*p14*p13*p13+p13*p0*p0*p87*p87+p14*p88*p88*p0*p0+p0*p16*p16*p15*p15+p15*p0*p0*p89*p89+p16*p90*p90*p0*p0+p0*p18*p18*p17*p17+p17*p0*p0*p91*p91+p18*p92*p92*p0*p0+p0*p20*p20*p19*p19+p19*p0*p0*p93*p93+p20*p94*p94*p0*p0)/18;
     AVcorr[989]+=(p0*p10*p9+p9*p0*p83+p10*p84*p0+p0*p12*p11+p11*p0*p85+p12*p86*p0+p0*p14*p13+p13*p0*p87+p14*p88*p0+p0*p16*p15+p15*p0*p89+p16*p90*p0+p0*p18*p17+p17*p0*p91+p18*p92*p0+p0*p20*p19+p19*p0*p93+p20*p94*p0)/18;
     AVcorr[997]+=(p45*p0*p0*p89+p49*p90*p90*p0+p49*p0*p0*p90+p45*p89*p89*p0+p47*p0*p0*p93+p49*p94*p94*p0+p49*p0*p0*p94+p47*p93*p93*p0+p39*p0*p0*p85+p50*p86*p86*p0+p50*p0*p0*p86+p39*p85*p85*p0+p43*p0*p0*p87+p50*p88*p88*p0+p50*p0*p0*p88+p43*p87*p87*p0+p45*p0*p0*p91+p47*p92*p92*p0+p47*p0*p0*p92+p45*p91*p91*p0+p37*p0*p0*p83+p48*p84*p84*p0+p48*p0*p0*p84+p37*p83*p83*p0+p41*p0*p0*p87+p48*p88*p88*p0+p48*p0*p0*p88+p41*p87*p87*p0+p33*p0*p0*p83+p46*p84*p84*p0+p46*p0*p0*p84+p33*p83*p83*p0+p35*p0*p0*p85+p46*p86*p86*p0+p46*p0*p0*p86+p35*p85*p85*p0+p39*p0*p0*p91+p43*p92*p92*p0+p43*p0*p0*p92+p39*p91*p91*p0+p31*p0*p0*p83+p44*p84*p84*p0+p44*p0*p0*p84+p31*p83*p83*p0+p42*p0*p0*p93+p44*p94*p94*p0+p44*p0*p0*p94+p42*p93*p93*p0+p37*p0*p0*p89+p41*p90*p90*p0+p41*p0*p0*p90+p37*p89*p89*p0+p31*p0*p0*p85+p42*p86*p86*p0+p42*p0*p0*p86+p31*p85*p85*p0+p29*p0*p0*p83+p40*p84*p84*p0+p40*p0*p0*p84+p29*p83*p83*p0+p36*p0*p0*p89+p40*p90*p90*p0+p40*p0*p0*p90+p36*p89*p89*p0+p27*p0*p0*p85+p38*p86*p86*p0+p38*p0*p0*p86+p27*p85*p85*p0+p34*p0*p0*p91+p38*p92*p92*p0+p38*p0*p0*p92+p34*p91*p91*p0+p33*p0*p0*p93+p35*p94*p94*p0+p35*p0*p0*p94+p33*p93*p93*p0+p29*p0*p0*p87+p36*p88*p88*p0+p36*p0*p0*p88+p29*p87*p87*p0+p27*p0*p0*p87+p34*p88*p88*p0+p34*p0*p0*p88+p27*p87*p87*p0+p28*p0*p0*p89+p32*p90*p90*p0+p32*p0*p0*p90+p28*p89*p89*p0+p30*p0*p0*p91+p32*p92*p92*p0+p32*p0*p0*p92+p30*p91*p91*p0+p28*p0*p0*p93+p30*p94*p94*p0+p30*p0*p0*p94+p28*p93*p93*p0)/144;
     AVcorr[1062]+=(p0*p0*p9*p63+p10*p10*p0*p86+p64*p64*p85*p0+p0*p0*p9*p67+p10*p10*p0*p88+p68*p68*p87*p0+p0*p0*p9*p68+p10*p10*p0*p90+p67*p67*p89*p0+p0*p0*p9*p64+p10*p10*p0*p94+p63*p63*p93*p0+p0*p0*p11*p58+p12*p12*p0*p84+p57*p57*p83*p0+p0*p0*p11*p66+p12*p12*p0*p88+p65*p65*p87*p0+p0*p0*p11*p65+p12*p12*p0*p92+p66*p66*p91*p0+p0*p0*p11*p57+p12*p12*p0*p93+p58*p58*p94*p0+p0*p0*p13*p56+p14*p14*p0*p84+p55*p55*p83*p0+p0*p0*p13*p62+p14*p14*p0*p86+p61*p61*p85*p0+p0*p0*p13*p61+p14*p14*p0*p91+p62*p62*p92*p0+p0*p0*p13*p55+p14*p14*p0*p89+p56*p56*p90*p0+p0*p0*p15*p54+p16*p16*p0*p84+p53*p53*p83*p0+p0*p0*p15*p74+p16*p16*p0*p92+p73*p73*p91*p0+p0*p0*p15*p73+p16*p16*p0*p94+p74*p74*p93*p0+p0*p0*p15*p53+p16*p16*p0*p87+p54*p54*p88*p0+p0*p0*p17*p60+p18*p18*p0*p86+p59*p59*p85*p0+p0*p0*p17*p72+p18*p18*p0*p90+p71*p71*p89*p0+p0*p0*p17*p71+p18*p18*p0*p93+p72*p72*p94*p0+p0*p0*p17*p59+p18*p18*p0*p87+p60*p60*p88*p0+p0*p0*p19*p52+p20*p20*p0*p84+p51*p51*p83*p0+p0*p0*p19*p70+p20*p20*p0*p90+p69*p69*p89*p0+p0*p0*p19*p69+p20*p20*p0*p91+p70*p70*p92*p0+p0*p0*p19*p51+p20*p20*p0*p85+p52*p52*p86*p0+p0*p0*p20*p52+p19*p19*p0*p86+p51*p51*p85*p0+p0*p0*p20*p70+p19*p19*p0*p92+p69*p69*p91*p0+p0*p0*p20*p69+p19*p19*p0*p89+p70*p70*p90*p0+p0*p0*p20*p51+p19*p19*p0*p83+p52*p52*p84*p0+p0*p0*p18*p60+p17*p17*p0*p88+p59*p59*p87*p0+p0*p0*p18*p72+p17*p17*p0*p94+p71*p71*p93*p0+p0*p0*p18*p71+p17*p17*p0*p89+p72*p72*p90*p0+p0*p0*p18*p59+p17*p17*p0*p85+p60*p60*p86*p0+p0*p0*p16*p54+p15*p15*p0*p88+p53*p53*p87*p0+p0*p0*p16*p74+p15*p15*p0*p93+p73*p73*p94*p0+p0*p0*p16*p73+p15*p15*p0*p91+p74*p74*p92*p0+p0*p0*p16*p53+p15*p15*p0*p83+p54*p54*p84*p0+p0*p0*p14*p56+p13*p13*p0*p90+p55*p55*p89*p0+p0*p0*p14*p62+p13*p13*p0*p92+p61*p61*p91*p0+p0*p0*p14*p61+p13*p13*p0*p85+p62*p62*p86*p0+p0*p0*p14*p55+p13*p13*p0*p83+p56*p56*p84*p0+p0*p0*p12*p58+p11*p11*p0*p94+p57*p57*p93*p0+p0*p0*p12*p66+p11*p11*p0*p91+p65*p65*p92*p0+p0*p0*p12*p65+p11*p11*p0*p87+p66*p66*p88*p0+p0*p0*p12*p57+p11*p11*p0*p83+p58*p58*p84*p0+p0*p0*p10*p63+p9*p9*p0*p93+p64*p64*p94*p0+p0*p0*p10*p67+p9*p9*p0*p89+p68*p68*p90*p0+p0*p0*p10*p68+p9*p9*p0*p87+p67*p67*p88*p0+p0*p0*p10*p64+p9*p9*p0*p85+p63*p63*p86*p0)/144;
     AVcorr[1063]+=(p0*p9*p9*p63+p10*p0*p0*p86+p64*p85*p85*p0+p0*p9*p9*p67+p10*p0*p0*p88+p68*p87*p87*p0+p0*p9*p9*p68+p10*p0*p0*p90+p67*p89*p89*p0+p0*p9*p9*p64+p10*p0*p0*p94+p63*p93*p93*p0+p0*p11*p11*p58+p12*p0*p0*p84+p57*p83*p83*p0+p0*p11*p11*p66+p12*p0*p0*p88+p65*p87*p87*p0+p0*p11*p11*p65+p12*p0*p0*p92+p66*p91*p91*p0+p0*p11*p11*p57+p12*p0*p0*p93+p58*p94*p94*p0+p0*p13*p13*p56+p14*p0*p0*p84+p55*p83*p83*p0+p0*p13*p13*p62+p14*p0*p0*p86+p61*p85*p85*p0+p0*p13*p13*p61+p14*p0*p0*p91+p62*p92*p92*p0+p0*p13*p13*p55+p14*p0*p0*p89+p56*p90*p90*p0+p0*p15*p15*p54+p16*p0*p0*p84+p53*p83*p83*p0+p0*p15*p15*p74+p16*p0*p0*p92+p73*p91*p91*p0+p0*p15*p15*p73+p16*p0*p0*p94+p74*p93*p93*p0+p0*p15*p15*p53+p16*p0*p0*p87+p54*p88*p88*p0+p0*p17*p17*p60+p18*p0*p0*p86+p59*p85*p85*p0+p0*p17*p17*p72+p18*p0*p0*p90+p71*p89*p89*p0+p0*p17*p17*p71+p18*p0*p0*p93+p72*p94*p94*p0+p0*p17*p17*p59+p18*p0*p0*p87+p60*p88*p88*p0+p0*p19*p19*p52+p20*p0*p0*p84+p51*p83*p83*p0+p0*p19*p19*p70+p20*p0*p0*p90+p69*p89*p89*p0+p0*p19*p19*p69+p20*p0*p0*p91+p70*p92*p92*p0+p0*p19*p19*p51+p20*p0*p0*p85+p52*p86*p86*p0+p0*p20*p20*p52+p19*p0*p0*p86+p51*p85*p85*p0+p0*p20*p20*p70+p19*p0*p0*p92+p69*p91*p91*p0+p0*p20*p20*p69+p19*p0*p0*p89+p70*p90*p90*p0+p0*p20*p20*p51+p19*p0*p0*p83+p52*p84*p84*p0+p0*p18*p18*p60+p17*p0*p0*p88+p59*p87*p87*p0+p0*p18*p18*p72+p17*p0*p0*p94+p71*p93*p93*p0+p0*p18*p18*p71+p17*p0*p0*p89+p72*p90*p90*p0+p0*p18*p18*p59+p17*p0*p0*p85+p60*p86*p86*p0+p0*p16*p16*p54+p15*p0*p0*p88+p53*p87*p87*p0+p0*p16*p16*p74+p15*p0*p0*p93+p73*p94*p94*p0+p0*p16*p16*p73+p15*p0*p0*p91+p74*p92*p92*p0+p0*p16*p16*p53+p15*p0*p0*p83+p54*p84*p84*p0+p0*p14*p14*p56+p13*p0*p0*p90+p55*p89*p89*p0+p0*p14*p14*p62+p13*p0*p0*p92+p61*p91*p91*p0+p0*p14*p14*p61+p13*p0*p0*p85+p62*p86*p86*p0+p0*p14*p14*p55+p13*p0*p0*p83+p56*p84*p84*p0+p0*p12*p12*p58+p11*p0*p0*p94+p57*p93*p93*p0+p0*p12*p12*p66+p11*p0*p0*p91+p65*p92*p92*p0+p0*p12*p12*p65+p11*p0*p0*p87+p66*p88*p88*p0+p0*p12*p12*p57+p11*p0*p0*p83+p58*p84*p84*p0+p0*p10*p10*p63+p9*p0*p0*p93+p64*p94*p94*p0+p0*p10*p10*p67+p9*p0*p0*p89+p68*p90*p90*p0+p0*p10*p10*p68+p9*p0*p0*p87+p67*p88*p88*p0+p0*p10*p10*p64+p9*p0*p0*p85+p63*p86*p86*p0)/144;
     AVcorr[1065]+=(p0*p9*p63*p63+p10*p0*p86*p86+p64*p85*p0*p0+p0*p9*p67*p67+p10*p0*p88*p88+p68*p87*p0*p0+p0*p9*p68*p68+p10*p0*p90*p90+p67*p89*p0*p0+p0*p9*p64*p64+p10*p0*p94*p94+p63*p93*p0*p0+p0*p11*p58*p58+p12*p0*p84*p84+p57*p83*p0*p0+p0*p11*p66*p66+p12*p0*p88*p88+p65*p87*p0*p0+p0*p11*p65*p65+p12*p0*p92*p92+p66*p91*p0*p0+p0*p11*p57*p57+p12*p0*p93*p93+p58*p94*p0*p0+p0*p13*p56*p56+p14*p0*p84*p84+p55*p83*p0*p0+p0*p13*p62*p62+p14*p0*p86*p86+p61*p85*p0*p0+p0*p13*p61*p61+p14*p0*p91*p91+p62*p92*p0*p0+p0*p13*p55*p55+p14*p0*p89*p89+p56*p90*p0*p0+p0*p15*p54*p54+p16*p0*p84*p84+p53*p83*p0*p0+p0*p15*p74*p74+p16*p0*p92*p92+p73*p91*p0*p0+p0*p15*p73*p73+p16*p0*p94*p94+p74*p93*p0*p0+p0*p15*p53*p53+p16*p0*p87*p87+p54*p88*p0*p0+p0*p17*p60*p60+p18*p0*p86*p86+p59*p85*p0*p0+p0*p17*p72*p72+p18*p0*p90*p90+p71*p89*p0*p0+p0*p17*p71*p71+p18*p0*p93*p93+p72*p94*p0*p0+p0*p17*p59*p59+p18*p0*p87*p87+p60*p88*p0*p0+p0*p19*p52*p52+p20*p0*p84*p84+p51*p83*p0*p0+p0*p19*p70*p70+p20*p0*p90*p90+p69*p89*p0*p0+p0*p19*p69*p69+p20*p0*p91*p91+p70*p92*p0*p0+p0*p19*p51*p51+p20*p0*p85*p85+p52*p86*p0*p0+p0*p20*p52*p52+p19*p0*p86*p86+p51*p85*p0*p0+p0*p20*p70*p70+p19*p0*p92*p92+p69*p91*p0*p0+p0*p20*p69*p69+p19*p0*p89*p89+p70*p90*p0*p0+p0*p20*p51*p51+p19*p0*p83*p83+p52*p84*p0*p0+p0*p18*p60*p60+p17*p0*p88*p88+p59*p87*p0*p0+p0*p18*p72*p72+p17*p0*p94*p94+p71*p93*p0*p0+p0*p18*p71*p71+p17*p0*p89*p89+p72*p90*p0*p0+p0*p18*p59*p59+p17*p0*p85*p85+p60*p86*p0*p0+p0*p16*p54*p54+p15*p0*p88*p88+p53*p87*p0*p0+p0*p16*p74*p74+p15*p0*p93*p93+p73*p94*p0*p0+p0*p16*p73*p73+p15*p0*p91*p91+p74*p92*p0*p0+p0*p16*p53*p53+p15*p0*p83*p83+p54*p84*p0*p0+p0*p14*p56*p56+p13*p0*p90*p90+p55*p89*p0*p0+p0*p14*p62*p62+p13*p0*p92*p92+p61*p91*p0*p0+p0*p14*p61*p61+p13*p0*p85*p85+p62*p86*p0*p0+p0*p14*p55*p55+p13*p0*p83*p83+p56*p84*p0*p0+p0*p12*p58*p58+p11*p0*p94*p94+p57*p93*p0*p0+p0*p12*p66*p66+p11*p0*p91*p91+p65*p92*p0*p0+p0*p12*p65*p65+p11*p0*p87*p87+p66*p88*p0*p0+p0*p12*p57*p57+p11*p0*p83*p83+p58*p84*p0*p0+p0*p10*p63*p63+p9*p0*p93*p93+p64*p94*p0*p0+p0*p10*p67*p67+p9*p0*p89*p89+p68*p90*p0*p0+p0*p10*p68*p68+p9*p0*p87*p87+p67*p88*p0*p0+p0*p10*p64*p64+p9*p0*p85*p85+p63*p86*p0*p0)/144;
     AVcorr[1068]+=(p0*p0*p9*p9*p63*p63+p10*p10*p0*p0*p86*p86+p64*p64*p85*p85*p0*p0+p0*p0*p9*p9*p67*p67+p10*p10*p0*p0*p88*p88+p68*p68*p87*p87*p0*p0+p0*p0*p9*p9*p68*p68+p10*p10*p0*p0*p90*p90+p67*p67*p89*p89*p0*p0+p0*p0*p9*p9*p64*p64+p10*p10*p0*p0*p94*p94+p63*p63*p93*p93*p0*p0+p0*p0*p11*p11*p58*p58+p12*p12*p0*p0*p84*p84+p57*p57*p83*p83*p0*p0+p0*p0*p11*p11*p66*p66+p12*p12*p0*p0*p88*p88+p65*p65*p87*p87*p0*p0+p0*p0*p11*p11*p65*p65+p12*p12*p0*p0*p92*p92+p66*p66*p91*p91*p0*p0+p0*p0*p11*p11*p57*p57+p12*p12*p0*p0*p93*p93+p58*p58*p94*p94*p0*p0+p0*p0*p13*p13*p56*p56+p14*p14*p0*p0*p84*p84+p55*p55*p83*p83*p0*p0+p0*p0*p13*p13*p62*p62+p14*p14*p0*p0*p86*p86+p61*p61*p85*p85*p0*p0+p0*p0*p13*p13*p61*p61+p14*p14*p0*p0*p91*p91+p62*p62*p92*p92*p0*p0+p0*p0*p13*p13*p55*p55+p14*p14*p0*p0*p89*p89+p56*p56*p90*p90*p0*p0+p0*p0*p15*p15*p54*p54+p16*p16*p0*p0*p84*p84+p53*p53*p83*p83*p0*p0+p0*p0*p15*p15*p74*p74+p16*p16*p0*p0*p92*p92+p73*p73*p91*p91*p0*p0+p0*p0*p15*p15*p73*p73+p16*p16*p0*p0*p94*p94+p74*p74*p93*p93*p0*p0+p0*p0*p15*p15*p53*p53+p16*p16*p0*p0*p87*p87+p54*p54*p88*p88*p0*p0+p0*p0*p17*p17*p60*p60+p18*p18*p0*p0*p86*p86+p59*p59*p85*p85*p0*p0+p0*p0*p17*p17*p72*p72+p18*p18*p0*p0*p90*p90+p71*p71*p89*p89*p0*p0+p0*p0*p17*p17*p71*p71+p18*p18*p0*p0*p93*p93+p72*p72*p94*p94*p0*p0+p0*p0*p17*p17*p59*p59+p18*p18*p0*p0*p87*p87+p60*p60*p88*p88*p0*p0+p0*p0*p19*p19*p52*p52+p20*p20*p0*p0*p84*p84+p51*p51*p83*p83*p0*p0+p0*p0*p19*p19*p70*p70+p20*p20*p0*p0*p90*p90+p69*p69*p89*p89*p0*p0+p0*p0*p19*p19*p69*p69+p20*p20*p0*p0*p91*p91+p70*p70*p92*p92*p0*p0+p0*p0*p19*p19*p51*p51+p20*p20*p0*p0*p85*p85+p52*p52*p86*p86*p0*p0+p0*p0*p20*p20*p52*p52+p19*p19*p0*p0*p86*p86+p51*p51*p85*p85*p0*p0+p0*p0*p20*p20*p70*p70+p19*p19*p0*p0*p92*p92+p69*p69*p91*p91*p0*p0+p0*p0*p20*p20*p69*p69+p19*p19*p0*p0*p89*p89+p70*p70*p90*p90*p0*p0+p0*p0*p20*p20*p51*p51+p19*p19*p0*p0*p83*p83+p52*p52*p84*p84*p0*p0+p0*p0*p18*p18*p60*p60+p17*p17*p0*p0*p88*p88+p59*p59*p87*p87*p0*p0+p0*p0*p18*p18*p72*p72+p17*p17*p0*p0*p94*p94+p71*p71*p93*p93*p0*p0+p0*p0*p18*p18*p71*p71+p17*p17*p0*p0*p89*p89+p72*p72*p90*p90*p0*p0+p0*p0*p18*p18*p59*p59+p17*p17*p0*p0*p85*p85+p60*p60*p86*p86*p0*p0+p0*p0*p16*p16*p54*p54+p15*p15*p0*p0*p88*p88+p53*p53*p87*p87*p0*p0+p0*p0*p16*p16*p74*p74+p15*p15*p0*p0*p93*p93+p73*p73*p94*p94*p0*p0+p0*p0*p16*p16*p73*p73+p15*p15*p0*p0*p91*p91+p74*p74*p92*p92*p0*p0+p0*p0*p16*p16*p53*p53+p15*p15*p0*p0*p83*p83+p54*p54*p84*p84*p0*p0+p0*p0*p14*p14*p56*p56+p13*p13*p0*p0*p90*p90+p55*p55*p89*p89*p0*p0+p0*p0*p14*p14*p62*p62+p13*p13*p0*p0*p92*p92+p61*p61*p91*p91*p0*p0+p0*p0*p14*p14*p61*p61+p13*p13*p0*p0*p85*p85+p62*p62*p86*p86*p0*p0+p0*p0*p14*p14*p55*p55+p13*p13*p0*p0*p83*p83+p56*p56*p84*p84*p0*p0+p0*p0*p12*p12*p58*p58+p11*p11*p0*p0*p94*p94+p57*p57*p93*p93*p0*p0+p0*p0*p12*p12*p66*p66+p11*p11*p0*p0*p91*p91+p65*p65*p92*p92*p0*p0+p0*p0*p12*p12*p65*p65+p11*p11*p0*p0*p87*p87+p66*p66*p88*p88*p0*p0+p0*p0*p12*p12*p57*p57+p11*p11*p0*p0*p83*p83+p58*p58*p84*p84*p0*p0+p0*p0*p10*p10*p63*p63+p9*p9*p0*p0*p93*p93+p64*p64*p94*p94*p0*p0+p0*p0*p10*p10*p67*p67+p9*p9*p0*p0*p89*p89+p68*p68*p90*p90*p0*p0+p0*p0*p10*p10*p68*p68+p9*p9*p0*p0*p87*p87+p67*p67*p88*p88*p0*p0+p0*p0*p10*p10*p64*p64+p9*p9*p0*p0*p85*p85+p63*p63*p86*p86*p0*p0)/144;
     AVcorr[1069]+=(p0*p9*p63+p10*p0*p86+p64*p85*p0+p0*p9*p67+p10*p0*p88+p68*p87*p0+p0*p9*p68+p10*p0*p90+p67*p89*p0+p0*p9*p64+p10*p0*p94+p63*p93*p0+p0*p11*p58+p12*p0*p84+p57*p83*p0+p0*p11*p66+p12*p0*p88+p65*p87*p0+p0*p11*p65+p12*p0*p92+p66*p91*p0+p0*p11*p57+p12*p0*p93+p58*p94*p0+p0*p13*p56+p14*p0*p84+p55*p83*p0+p0*p13*p62+p14*p0*p86+p61*p85*p0+p0*p13*p61+p14*p0*p91+p62*p92*p0+p0*p13*p55+p14*p0*p89+p56*p90*p0+p0*p15*p54+p16*p0*p84+p53*p83*p0+p0*p15*p74+p16*p0*p92+p73*p91*p0+p0*p15*p73+p16*p0*p94+p74*p93*p0+p0*p15*p53+p16*p0*p87+p54*p88*p0+p0*p17*p60+p18*p0*p86+p59*p85*p0+p0*p17*p72+p18*p0*p90+p71*p89*p0+p0*p17*p71+p18*p0*p93+p72*p94*p0+p0*p17*p59+p18*p0*p87+p60*p88*p0+p0*p19*p52+p20*p0*p84+p51*p83*p0+p0*p19*p70+p20*p0*p90+p69*p89*p0+p0*p19*p69+p20*p0*p91+p70*p92*p0+p0*p19*p51+p20*p0*p85+p52*p86*p0+p0*p20*p52+p19*p0*p86+p51*p85*p0+p0*p20*p70+p19*p0*p92+p69*p91*p0+p0*p20*p69+p19*p0*p89+p70*p90*p0+p0*p20*p51+p19*p0*p83+p52*p84*p0+p0*p18*p60+p17*p0*p88+p59*p87*p0+p0*p18*p72+p17*p0*p94+p71*p93*p0+p0*p18*p71+p17*p0*p89+p72*p90*p0+p0*p18*p59+p17*p0*p85+p60*p86*p0+p0*p16*p54+p15*p0*p88+p53*p87*p0+p0*p16*p74+p15*p0*p93+p73*p94*p0+p0*p16*p73+p15*p0*p91+p74*p92*p0+p0*p16*p53+p15*p0*p83+p54*p84*p0+p0*p14*p56+p13*p0*p90+p55*p89*p0+p0*p14*p62+p13*p0*p92+p61*p91*p0+p0*p14*p61+p13*p0*p85+p62*p86*p0+p0*p14*p55+p13*p0*p83+p56*p84*p0+p0*p12*p58+p11*p0*p94+p57*p93*p0+p0*p12*p66+p11*p0*p91+p65*p92*p0+p0*p12*p65+p11*p0*p87+p66*p88*p0+p0*p12*p57+p11*p0*p83+p58*p84*p0+p0*p10*p63+p9*p0*p93+p64*p94*p0+p0*p10*p67+p9*p0*p89+p68*p90*p0+p0*p10*p68+p9*p0*p87+p67*p88*p0+p0*p10*p64+p9*p0*p85+p63*p86*p0)/144;
     AVcorr[1070]+=(p0*p0*p61*p94+p62*p62*p0*p56+p93*p93*p55*p0+p0*p0*p55*p93+p56*p56*p0*p62+p94*p94*p61*p0+p0*p0*p65*p90+p66*p66*p0*p58+p89*p89*p57*p0+p0*p0*p57*p89+p58*p58*p0*p66+p90*p90*p65*p0+p0*p0*p69*p88+p70*p70*p0*p52+p87*p87*p51*p0+p0*p0*p51*p87+p52*p52*p0*p70+p88*p88*p69*p0+p0*p0*p74*p86+p73*p73*p0*p54+p85*p85*p53*p0+p0*p0*p53*p85+p54*p54*p0*p73+p86*p86*p74*p0+p0*p0*p68*p92+p67*p67*p0*p63+p91*p91*p64*p0+p0*p0*p64*p91+p63*p63*p0*p67+p92*p92*p68*p0+p0*p0*p72*p84+p71*p71*p0*p60+p83*p83*p59*p0+p0*p0*p59*p83+p60*p60*p0*p71+p84*p84*p72*p0+p0*p0*p63*p92+p64*p64*p0*p68+p91*p91*p67*p0+p0*p0*p67*p91+p68*p68*p0*p64+p92*p92*p63*p0+p0*p0*p60*p84+p59*p59*p0*p72+p83*p83*p71*p0+p0*p0*p71*p83+p72*p72*p0*p59+p84*p84*p60*p0+p0*p0*p58*p90+p57*p57*p0*p65+p89*p89*p66*p0+p0*p0*p66*p89+p65*p65*p0*p57+p90*p90*p58*p0+p0*p0*p54*p86+p53*p53*p0*p74+p85*p85*p73*p0+p0*p0*p73*p85+p74*p74*p0*p53+p86*p86*p54*p0+p0*p0*p56*p94+p55*p55*p0*p61+p93*p93*p62*p0+p0*p0*p62*p93+p61*p61*p0*p55+p94*p94*p56*p0+p0*p0*p52*p88+p51*p51*p0*p69+p87*p87*p70*p0+p0*p0*p70*p87+p69*p69*p0*p51+p88*p88*p52*p0)/72;
     AVcorr[1071]+=(p0*p61*p61*p94+p62*p0*p0*p56+p93*p55*p55*p0+p0*p65*p65*p90+p66*p0*p0*p58+p89*p57*p57*p0+p0*p69*p69*p88+p70*p0*p0*p52+p87*p51*p51*p0+p0*p74*p74*p86+p73*p0*p0*p54+p85*p53*p53*p0+p0*p68*p68*p92+p67*p0*p0*p63+p91*p64*p64*p0+p0*p72*p72*p84+p71*p0*p0*p60+p83*p59*p59*p0+p0*p63*p63*p92+p64*p0*p0*p68+p91*p67*p67*p0+p0*p60*p60*p84+p59*p0*p0*p72+p83*p71*p71*p0+p0*p58*p58*p90+p57*p0*p0*p65+p89*p66*p66*p0+p0*p54*p54*p86+p53*p0*p0*p74+p85*p73*p73*p0+p0*p56*p56*p94+p55*p0*p0*p61+p93*p62*p62*p0+p0*p52*p52*p88+p51*p0*p0*p69+p87*p70*p70*p0)/36;
     AVcorr[1073]+=(p0*p0*p61*p94*p94+p62*p62*p0*p56*p56+p93*p93*p55*p0*p0+p0*p0*p65*p90*p90+p66*p66*p0*p58*p58+p89*p89*p57*p0*p0+p0*p0*p69*p88*p88+p70*p70*p0*p52*p52+p87*p87*p51*p0*p0+p0*p0*p74*p86*p86+p73*p73*p0*p54*p54+p85*p85*p53*p0*p0+p0*p0*p68*p92*p92+p67*p67*p0*p63*p63+p91*p91*p64*p0*p0+p0*p0*p72*p84*p84+p71*p71*p0*p60*p60+p83*p83*p59*p0*p0+p0*p0*p63*p92*p92+p64*p64*p0*p68*p68+p91*p91*p67*p0*p0+p0*p0*p60*p84*p84+p59*p59*p0*p72*p72+p83*p83*p71*p0*p0+p0*p0*p58*p90*p90+p57*p57*p0*p65*p65+p89*p89*p66*p0*p0+p0*p0*p54*p86*p86+p53*p53*p0*p74*p74+p85*p85*p73*p0*p0+p0*p0*p56*p94*p94+p55*p55*p0*p61*p61+p93*p93*p62*p0*p0+p0*p0*p52*p88*p88+p51*p51*p0*p69*p69+p87*p87*p70*p0*p0)/36;
     AVcorr[1074]+=(p0*p0*p61*p61*p94*p94+p62*p62*p0*p0*p56*p56+p93*p93*p55*p55*p0*p0+p0*p0*p65*p65*p90*p90+p66*p66*p0*p0*p58*p58+p89*p89*p57*p57*p0*p0+p0*p0*p69*p69*p88*p88+p70*p70*p0*p0*p52*p52+p87*p87*p51*p51*p0*p0+p0*p0*p74*p74*p86*p86+p73*p73*p0*p0*p54*p54+p85*p85*p53*p53*p0*p0+p0*p0*p68*p68*p92*p92+p67*p67*p0*p0*p63*p63+p91*p91*p64*p64*p0*p0+p0*p0*p72*p72*p84*p84+p71*p71*p0*p0*p60*p60+p83*p83*p59*p59*p0*p0+p0*p0*p63*p63*p92*p92+p64*p64*p0*p0*p68*p68+p91*p91*p67*p67*p0*p0+p0*p0*p60*p60*p84*p84+p59*p59*p0*p0*p72*p72+p83*p83*p71*p71*p0*p0+p0*p0*p58*p58*p90*p90+p57*p57*p0*p0*p65*p65+p89*p89*p66*p66*p0*p0+p0*p0*p54*p54*p86*p86+p53*p53*p0*p0*p74*p74+p85*p85*p73*p73*p0*p0+p0*p0*p56*p56*p94*p94+p55*p55*p0*p0*p61*p61+p93*p93*p62*p62*p0*p0+p0*p0*p52*p52*p88*p88+p51*p51*p0*p0*p69*p69+p87*p87*p70*p70*p0*p0)/36;
     AVcorr[1075]+=(p0*p61*p94+p62*p0*p56+p93*p55*p0+p0*p65*p90+p66*p0*p58+p89*p57*p0+p0*p69*p88+p70*p0*p52+p87*p51*p0+p0*p74*p86+p73*p0*p54+p85*p53*p0+p0*p68*p92+p67*p0*p63+p91*p64*p0+p0*p72*p84+p71*p0*p60+p83*p59*p0+p0*p63*p92+p64*p0*p68+p91*p67*p0+p0*p60*p84+p59*p0*p72+p83*p71*p0+p0*p58*p90+p57*p0*p65+p89*p66*p0+p0*p54*p86+p53*p0*p74+p85*p73*p0+p0*p56*p94+p55*p0*p61+p93*p62*p0+p0*p52*p88+p51*p0*p69+p87*p70*p0)/36;
     AVcorr[1088]+=(p0*p0*p84*p86+p83*p83*p0*p93+p85*p85*p94*p0+p0*p0*p94*p85+p93*p93*p0*p83+p86*p86*p84*p0+p0*p0*p93*p83+p94*p94*p0*p85+p84*p84*p86*p0+p0*p0*p84*p88+p83*p83*p0*p89+p87*p87*p90*p0+p0*p0*p90*p87+p89*p89*p0*p83+p88*p88*p84*p0+p0*p0*p89*p83+p90*p90*p0*p87+p84*p84*p88*p0+p0*p0*p84*p90+p83*p83*p0*p87+p89*p89*p88*p0+p0*p0*p88*p89+p87*p87*p0*p83+p90*p90*p84*p0+p0*p0*p87*p83+p88*p88*p0*p89+p84*p84*p90*p0+p0*p0*p84*p94+p83*p83*p0*p85+p93*p93*p86*p0+p0*p0*p86*p93+p85*p85*p0*p83+p94*p94*p84*p0+p0*p0*p85*p83+p86*p86*p0*p93+p84*p84*p94*p0+p0*p0*p86*p88+p85*p85*p0*p91+p87*p87*p92*p0+p0*p0*p92*p87+p91*p91*p0*p85+p88*p88*p86*p0+p0*p0*p91*p85+p92*p92*p0*p87+p86*p86*p88*p0+p0*p0*p86*p92+p85*p85*p0*p87+p91*p91*p88*p0+p0*p0*p88*p91+p87*p87*p0*p85+p92*p92*p86*p0+p0*p0*p87*p85+p88*p88*p0*p91+p86*p86*p92*p0+p0*p0*p90*p92+p89*p89*p0*p93+p91*p91*p94*p0+p0*p0*p94*p91+p93*p93*p0*p89+p92*p92*p90*p0+p0*p0*p93*p89+p94*p94*p0*p91+p90*p90*p92*p0+p0*p0*p90*p94+p89*p89*p0*p91+p93*p93*p92*p0+p0*p0*p92*p93+p91*p91*p0*p89+p94*p94*p90*p0+p0*p0*p91*p89+p92*p92*p0*p93+p90*p90*p94*p0)/72;
     AVcorr[1089]+=(p0*p0*p84*p84*p86+p83*p83*p0*p0*p93+p85*p85*p94*p94*p0+p0*p0*p86*p86*p84+p85*p85*p0*p0*p94+p83*p83*p93*p93*p0+p0*p0*p94*p94*p85+p93*p93*p0*p0*p83+p86*p86*p84*p84*p0+p0*p0*p84*p84*p88+p83*p83*p0*p0*p89+p87*p87*p90*p90*p0+p0*p0*p88*p88*p84+p87*p87*p0*p0*p90+p83*p83*p89*p89*p0+p0*p0*p90*p90*p87+p89*p89*p0*p0*p83+p88*p88*p84*p84*p0+p0*p0*p84*p84*p90+p83*p83*p0*p0*p87+p89*p89*p88*p88*p0+p0*p0*p88*p88*p89+p87*p87*p0*p0*p83+p90*p90*p84*p84*p0+p0*p0*p90*p90*p84+p89*p89*p0*p0*p88+p83*p83*p87*p87*p0+p0*p0*p84*p84*p94+p83*p83*p0*p0*p85+p93*p93*p86*p86*p0+p0*p0*p86*p86*p93+p85*p85*p0*p0*p83+p94*p94*p84*p84*p0+p0*p0*p94*p94*p84+p93*p93*p0*p0*p86+p83*p83*p85*p85*p0+p0*p0*p86*p86*p88+p85*p85*p0*p0*p91+p87*p87*p92*p92*p0+p0*p0*p92*p92*p87+p91*p91*p0*p0*p85+p88*p88*p86*p86*p0+p0*p0*p88*p88*p86+p87*p87*p0*p0*p92+p85*p85*p91*p91*p0+p0*p0*p86*p86*p92+p85*p85*p0*p0*p87+p91*p91*p88*p88*p0+p0*p0*p92*p92*p86+p91*p91*p0*p0*p88+p85*p85*p87*p87*p0+p0*p0*p88*p88*p91+p87*p87*p0*p0*p85+p92*p92*p86*p86*p0+p0*p0*p90*p90*p92+p89*p89*p0*p0*p93+p91*p91*p94*p94*p0+p0*p0*p92*p92*p90+p91*p91*p0*p0*p94+p89*p89*p93*p93*p0+p0*p0*p94*p94*p91+p93*p93*p0*p0*p89+p92*p92*p90*p90*p0+p0*p0*p90*p90*p94+p89*p89*p0*p0*p91+p93*p93*p92*p92*p0+p0*p0*p92*p92*p93+p91*p91*p0*p0*p89+p94*p94*p90*p90*p0+p0*p0*p94*p94*p90+p93*p93*p0*p0*p92+p89*p89*p91*p91*p0)/72;
     AVcorr[1090]+=(p0*p0*p84*p84*p86*p86+p83*p83*p0*p0*p93*p93+p85*p85*p94*p94*p0*p0+p0*p0*p84*p84*p88*p88+p83*p83*p0*p0*p89*p89+p87*p87*p90*p90*p0*p0+p0*p0*p84*p84*p90*p90+p83*p83*p0*p0*p87*p87+p89*p89*p88*p88*p0*p0+p0*p0*p84*p84*p94*p94+p83*p83*p0*p0*p85*p85+p93*p93*p86*p86*p0*p0+p0*p0*p86*p86*p88*p88+p85*p85*p0*p0*p91*p91+p87*p87*p92*p92*p0*p0+p0*p0*p86*p86*p92*p92+p85*p85*p0*p0*p87*p87+p91*p91*p88*p88*p0*p0+p0*p0*p90*p90*p92*p92+p89*p89*p0*p0*p93*p93+p91*p91*p94*p94*p0*p0+p0*p0*p90*p90*p94*p94+p89*p89*p0*p0*p91*p91+p93*p93*p92*p92*p0*p0)/24;
     AVcorr[1135]+=(p0*p0*p9*p19*p15+p10*p10*p0*p12*p14+p20*p20*p11*p0*p17+p16*p16*p13*p18*p0+p0*p0*p11*p20*p17+p12*p12*p0*p10*p14+p19*p19*p9*p0*p15+p18*p18*p13*p16*p0+p0*p0*p13*p16*p18+p14*p14*p0*p10*p12+p15*p15*p9*p0*p19+p17*p17*p11*p20*p0+p0*p0*p14*p10*p12+p13*p13*p0*p16*p18+p9*p9*p15*p0*p19+p11*p11*p17*p20*p0+p0*p0*p9*p13*p11+p10*p10*p0*p16*p20+p14*p14*p15*p0*p17+p12*p12*p19*p18*p0+p0*p0*p15*p14*p17+p16*p16*p0*p10*p20+p13*p13*p9*p0*p11+p18*p18*p19*p12*p0+p0*p0*p19*p12*p18+p20*p20*p0*p10*p16+p11*p11*p9*p0*p13+p17*p17*p15*p14*p0+p0*p0*p20*p10*p16+p19*p19*p0*p12*p18+p9*p9*p11*p0*p13+p15*p15*p17*p14*p0)/32;
     AVcorr[1137]+=(p0*p0*p9*p9*p19*p19*p15+p10*p10*p0*p0*p12*p12*p14+p20*p20*p11*p11*p0*p0*p17+p16*p16*p13*p13*p18*p18*p0+p0*p0*p9*p9*p15*p15*p19+p10*p10*p0*p0*p14*p14*p12+p16*p16*p13*p13*p0*p0*p18+p20*p20*p11*p11*p17*p17*p0+p0*p0*p11*p11*p17*p17*p20+p12*p12*p0*p0*p14*p14*p10+p18*p18*p13*p13*p0*p0*p16+p19*p19*p9*p9*p15*p15*p0+p0*p0*p15*p15*p19*p19*p9+p16*p16*p0*p0*p18*p18*p13+p20*p20*p17*p17*p0*p0*p11+p10*p10*p14*p14*p12*p12*p0+p0*p0*p9*p9*p13*p13*p11+p10*p10*p0*p0*p16*p16*p20+p14*p14*p15*p15*p0*p0*p17+p12*p12*p19*p19*p18*p18*p0+p0*p0*p9*p9*p11*p11*p13+p10*p10*p0*p0*p20*p20*p16+p12*p12*p19*p19*p0*p0*p18+p14*p14*p15*p15*p17*p17*p0+p0*p0*p11*p11*p13*p13*p9+p12*p12*p0*p0*p18*p18*p19+p14*p14*p17*p17*p0*p0*p15+p10*p10*p20*p20*p16*p16*p0+p0*p0*p15*p15*p17*p17*p14+p16*p16*p0*p0*p20*p20*p10+p18*p18*p19*p19*p0*p0*p12+p13*p13*p9*p9*p11*p11*p0)/32;
     AVcorr[1138]+=(p0*p0*p9*p9*p19*p19*p15*p15+p10*p10*p0*p0*p12*p12*p14*p14+p20*p20*p11*p11*p0*p0*p17*p17+p16*p16*p13*p13*p18*p18*p0*p0+p0*p0*p9*p9*p13*p13*p11*p11+p10*p10*p0*p0*p16*p16*p20*p20+p14*p14*p15*p15*p0*p0*p17*p17+p12*p12*p19*p19*p18*p18*p0*p0)/8;
     return;
  }


  if(b == 1){
     l=index(i,j,k,1); 
     double p0=mcL[l]; 
     l=index(i,j,k,0); 
     double p1=mcL[l]; 
     l=index(i,j,k+1,0); 
     double p2=mcL[l]; 
     l=index(i,j+1,k,0); 
     double p3=mcL[l]; 
     l=index(i+1,j,k,0); 
     double p4=mcL[l]; 
     l=index(i,j,k-1,2); 
     double p5=mcL[l]; 
     l=index(i,j-1,k,2); 
     double p6=mcL[l]; 
     l=index(i-1,j-1,k,2); 
     double p7=mcL[l]; 
     l=index(i-1,j,k-1,2); 
     double p8=mcL[l]; 
     l=index(i,j-1,k-1,2); 
     double p9=mcL[l]; 
     l=index(i-1,j,k,2); 
     double p10=mcL[l]; 
     l=index(i+1,j-1,k-1,2); 
     double p11=mcL[l]; 
     l=index(i-1,j-1,k-1,2); 
     double p12=mcL[l]; 
     l=index(i-1,j-1,k+1,2); 
     double p13=mcL[l]; 
     l=index(i-1,j+1,k-1,2); 
     double p14=mcL[l]; 
     l=index(i-1,j,k,0); 
     double p15=mcL[l]; 
     l=index(i-1,j,k+2,0); 
     double p16=mcL[l]; 
     l=index(i-1,j+2,k,0); 
     double p17=mcL[l]; 
     l=index(i,j-1,k,0); 
     double p18=mcL[l]; 
     l=index(i,j-1,k+2,0); 
     double p19=mcL[l]; 
     l=index(i,j,k-1,0); 
     double p20=mcL[l]; 
     l=index(i,j,k+2,0); 
     double p21=mcL[l]; 
     l=index(i,j+2,k-1,0); 
     double p22=mcL[l]; 
     l=index(i,j+2,k,0); 
     double p23=mcL[l]; 
     l=index(i+2,j-1,k,0); 
     double p24=mcL[l]; 
     l=index(i+2,j,k-1,0); 
     double p25=mcL[l]; 
     l=index(i+2,j,k,0); 
     double p26=mcL[l]; 
     l=index(i-2,j+1,k+1,0); 
     double p27=mcL[l]; 
     l=index(i+1,j-2,k+1,0); 
     double p28=mcL[l]; 
     l=index(i+1,j+1,k-2,0); 
     double p29=mcL[l]; 
     l=index(i+1,j+1,k+1,0); 
     double p30=mcL[l]; 
     l=index(i+2,j,k,1); 
     double p31=mcL[l]; 
     l=index(i-2,j,k,1); 
     double p32=mcL[l]; 
     l=index(i+2,j,k-2,1); 
     double p33=mcL[l]; 
     l=index(i-2,j,k+2,1); 
     double p34=mcL[l]; 
     l=index(i+2,j-2,k,1); 
     double p35=mcL[l]; 
     l=index(i-2,j+2,k,1); 
     double p36=mcL[l]; 
     l=index(i,j+2,k,1); 
     double p37=mcL[l]; 
     l=index(i,j-2,k,1); 
     double p38=mcL[l]; 
     l=index(i,j+2,k-2,1); 
     double p39=mcL[l]; 
     l=index(i,j-2,k+2,1); 
     double p40=mcL[l]; 
     l=index(i,j,k+2,1); 
     double p41=mcL[l]; 
     l=index(i,j,k-2,1); 
     double p42=mcL[l]; 
     l=index(i,j+1,k+1,0); 
     double p43=mcL[l]; 
     l=index(i-1,j+1,k+1,0); 
     double p44=mcL[l]; 
     l=index(i,j+1,k-1,0); 
     double p45=mcL[l]; 
     l=index(i-1,j+1,k,0); 
     double p46=mcL[l]; 
     l=index(i,j-1,k+1,0); 
     double p47=mcL[l]; 
     l=index(i-1,j,k+1,0); 
     double p48=mcL[l]; 
     l=index(i+1,j,k+1,0); 
     double p49=mcL[l]; 
     l=index(i+1,j-1,k+1,0); 
     double p50=mcL[l]; 
     l=index(i+1,j,k-1,0); 
     double p51=mcL[l]; 
     l=index(i+1,j-1,k,0); 
     double p52=mcL[l]; 
     l=index(i+1,j+1,k,0); 
     double p53=mcL[l]; 
     l=index(i+1,j+1,k-1,0); 
     double p54=mcL[l]; 
     l=index(i+1,j+2,k-1,0); 
     double p55=mcL[l]; 
     l=index(i+1,j-1,k+2,0); 
     double p56=mcL[l]; 
     l=index(i+1,j-1,k-1,0); 
     double p57=mcL[l]; 
     l=index(i+2,j+1,k-1,0); 
     double p58=mcL[l]; 
     l=index(i-1,j+1,k+2,0); 
     double p59=mcL[l]; 
     l=index(i-1,j+1,k-1,0); 
     double p60=mcL[l]; 
     l=index(i+2,j-1,k+1,0); 
     double p61=mcL[l]; 
     l=index(i-1,j+2,k+1,0); 
     double p62=mcL[l]; 
     l=index(i+2,j-1,k-1,0); 
     double p63=mcL[l]; 
     l=index(i-1,j+2,k-1,0); 
     double p64=mcL[l]; 
     l=index(i-1,j-1,k+1,0); 
     double p65=mcL[l]; 
     l=index(i-1,j-1,k+2,0); 
     double p66=mcL[l]; 

     AVcorr[0]+=1.0/3;
     AVcorr[1]+=(p0*p0)/2;
     AVcorr[2]+=(p0)/2;
     AVcorr[5]+=(p0*p0*p1+p0*p0*p2+p0*p0*p3+p0*p0*p4)/16;
     AVcorr[6]+=(p0*p1*p1+p0*p2*p2+p0*p3*p3+p0*p4*p4)/16;
     AVcorr[7]+=(p0*p0*p1*p1+p0*p0*p2*p2+p0*p0*p3*p3+p0*p0*p4*p4)/16;
     AVcorr[8]+=(p0*p1+p0*p2+p0*p3+p0*p4)/16;
     AVcorr[9]+=(p0*p0*p5+p5*p5*p0+p0*p0*p6+p6*p6*p0+p7*p7*p0+p0*p0*p7+p8*p8*p0+p0*p0*p8+p9*p9*p0+p0*p0*p9+p0*p0*p10+p10*p10*p0)/24;
     AVcorr[10]+=(p0*p0*p5*p5+p0*p0*p6*p6+p7*p7*p0*p0+p8*p8*p0*p0+p9*p9*p0*p0+p0*p0*p10*p10)/12;
     AVcorr[23]+=(p0*p0*p11*p11+p12*p12*p0*p0+p0*p0*p13*p13+p0*p0*p14*p14)/8;
     AVcorr[24]+=(p0*p11+p12*p0+p0*p13+p0*p14)/8;
     AVcorr[36]+=(p0*p0*p15*p15+p0*p0*p16*p16+p0*p0*p17*p17+p0*p0*p18*p18+p0*p0*p19*p19+p0*p0*p20*p20+p0*p0*p21*p21+p0*p0*p22*p22+p0*p0*p23*p23+p0*p0*p24*p24+p0*p0*p25*p25+p0*p0*p26*p26)/48;
     AVcorr[48]+=(p0*p0*p27+p0*p0*p28+p0*p0*p29+p0*p0*p30)/16;
     AVcorr[50]+=(p0*p0*p27*p27+p0*p0*p28*p28+p0*p0*p29*p29+p0*p0*p30*p30)/16;
     AVcorr[51]+=(p0*p27+p0*p28+p0*p29+p0*p30)/16;
     AVcorr[59]+=(p0*p0*p31+p32*p32*p0+p0*p0*p32+p31*p31*p0+p0*p0*p33+p34*p34*p0+p0*p0*p34+p33*p33*p0+p0*p0*p35+p36*p36*p0+p0*p0*p36+p35*p35*p0+p0*p0*p37+p38*p38*p0+p0*p0*p38+p37*p37*p0+p0*p0*p39+p40*p40*p0+p0*p0*p40+p39*p39*p0+p0*p0*p41+p42*p42*p0+p0*p0*p42+p41*p41*p0)/48;
     AVcorr[61]+=(p0*p31+p32*p0+p0*p33+p34*p0+p0*p35+p36*p0+p0*p37+p38*p0+p0*p39+p40*p0+p0*p41+p42*p0)/24;
     AVcorr[134]+=(p43*p44*p0*p0+p45*p46*p0*p0+p47*p48*p0*p0+p49*p50*p0*p0+p51*p52*p0*p0+p53*p54*p0*p0)/36;
     AVcorr[137]+=(p43*p44*p0+p45*p46*p0+p47*p48*p0+p49*p50*p0+p51*p52*p0+p53*p54*p0)/36;
     AVcorr[197]+=(p0*p4*p47*p47+p0*p4*p45*p45+p0*p4*p43*p43+p0*p3*p48*p48+p0*p3*p51*p51+p0*p3*p49*p49+p0*p2*p46*p46+p0*p2*p52*p52+p0*p1*p44*p44+p0*p1*p50*p50+p0*p2*p53*p53+p0*p1*p54*p54)/72;
     AVcorr[301]+=(p23*p23*p17*p0*p0+p17*p17*p23*p0*p0+p21*p21*p16*p0*p0+p16*p16*p21*p0*p0+p22*p22*p17*p0*p0+p17*p17*p22*p0*p0+p20*p20*p15*p0*p0+p15*p15*p20*p0*p0+p19*p19*p16*p0*p0+p16*p16*p19*p0*p0+p18*p18*p15*p0*p0+p15*p15*p18*p0*p0+p26*p26*p24*p0*p0+p24*p24*p26*p0*p0+p21*p21*p19*p0*p0+p19*p19*p21*p0*p0+p25*p25*p24*p0*p0+p24*p24*p25*p0*p0+p20*p20*p18*p0*p0+p18*p18*p20*p0*p0+p26*p26*p25*p0*p0+p25*p25*p26*p0*p0+p23*p23*p22*p0*p0+p22*p22*p23*p0*p0)/144;
     AVcorr[557]+=(p0*p53*p53*p21+p0*p49*p49*p23+p0*p54*p54*p20+p0*p51*p51*p22+p0*p50*p50*p18+p0*p52*p52*p19+p0*p43*p43*p26+p0*p45*p45*p25+p0*p47*p47*p24+p0*p44*p44*p15+p0*p46*p46*p16+p0*p48*p48*p17)/72;
     AVcorr[564]+=(p0*p0*p4*p18+p0*p0*p4*p20+p0*p0*p4*p19+p0*p0*p4*p21+p0*p0*p4*p22+p0*p0*p4*p23+p0*p0*p3*p15+p0*p0*p3*p20+p0*p0*p3*p16+p0*p0*p3*p21+p0*p0*p2*p15+p0*p0*p2*p18+p0*p0*p1*p16+p0*p0*p1*p19+p0*p0*p2*p17+p0*p0*p2*p23+p0*p0*p1*p17+p0*p0*p1*p22+p0*p0*p3*p25+p0*p0*p3*p26+p0*p0*p2*p24+p0*p0*p2*p26+p0*p0*p1*p24+p0*p0*p1*p25)/144;
     AVcorr[654]+=(p14*p14*p55*p0+p13*p13*p56*p0+p0*p0*p53*p13+p0*p0*p49*p14+p12*p12*p57*p0+p0*p0*p54*p12+p0*p0*p51*p14+p0*p0*p50*p12+p0*p0*p52*p13+p11*p11*p58*p0+p0*p0*p43*p11+p13*p13*p59*p0+p0*p0*p45*p11+p12*p12*p60*p0+p11*p11*p61*p0+p14*p14*p62*p0+p11*p11*p63*p0+p14*p14*p64*p0+p0*p0*p47*p11+p12*p12*p65*p0+p13*p13*p66*p0+p0*p0*p44*p12+p0*p0*p46*p13+p0*p0*p48*p14)/72;
     AVcorr[655]+=(p14*p55*p55*p0+p13*p56*p56*p0+p0*p53*p53*p13+p0*p49*p49*p14+p12*p57*p57*p0+p0*p54*p54*p12+p0*p51*p51*p14+p0*p50*p50*p12+p0*p52*p52*p13+p11*p58*p58*p0+p0*p43*p43*p11+p13*p59*p59*p0+p0*p45*p45*p11+p12*p60*p60*p0+p11*p61*p61*p0+p14*p62*p62*p0+p11*p63*p63*p0+p14*p64*p64*p0+p0*p47*p47*p11+p12*p65*p65*p0+p13*p66*p66*p0+p0*p44*p44*p12+p0*p46*p46*p13+p0*p48*p48*p14)/72;
     AVcorr[657]+=(p14*p55*p0*p0+p13*p56*p0*p0+p0*p53*p13*p13+p0*p49*p14*p14+p12*p57*p0*p0+p0*p54*p12*p12+p0*p51*p14*p14+p0*p50*p12*p12+p0*p52*p13*p13+p11*p58*p0*p0+p0*p43*p11*p11+p13*p59*p0*p0+p0*p45*p11*p11+p12*p60*p0*p0+p11*p61*p0*p0+p14*p62*p0*p0+p11*p63*p0*p0+p14*p64*p0*p0+p0*p47*p11*p11+p12*p65*p0*p0+p13*p66*p0*p0+p0*p44*p12*p12+p0*p46*p13*p13+p0*p48*p14*p14)/72;
     AVcorr[762]+=(p22*p22*p55*p0+p19*p19*p56*p0+p23*p23*p55*p0+p18*p18*p57*p0+p21*p21*p56*p0+p20*p20*p57*p0+p25*p25*p58*p0+p16*p16*p59*p0+p26*p26*p58*p0+p15*p15*p60*p0+p24*p24*p61*p0+p17*p17*p62*p0+p24*p24*p63*p0+p17*p17*p64*p0+p26*p26*p61*p0+p15*p15*p65*p0+p25*p25*p63*p0+p16*p16*p66*p0+p21*p21*p59*p0+p20*p20*p60*p0+p23*p23*p62*p0+p18*p18*p65*p0+p22*p22*p64*p0+p19*p19*p66*p0)/144;
     AVcorr[763]+=(p22*p55*p55*p0+p19*p56*p56*p0+p23*p55*p55*p0+p18*p57*p57*p0+p21*p56*p56*p0+p20*p57*p57*p0+p25*p58*p58*p0+p16*p59*p59*p0+p26*p58*p58*p0+p15*p60*p60*p0+p24*p61*p61*p0+p17*p62*p62*p0+p24*p63*p63*p0+p17*p64*p64*p0+p26*p61*p61*p0+p15*p65*p65*p0+p25*p63*p63*p0+p16*p66*p66*p0+p21*p59*p59*p0+p20*p60*p60*p0+p23*p62*p62*p0+p18*p65*p65*p0+p22*p64*p64*p0+p19*p66*p66*p0)/144;
     AVcorr[997]+=(p0*p21*p21*p26+p0*p26*p26*p21+p0*p23*p23*p26+p0*p26*p26*p23+p0*p20*p20*p25+p0*p25*p25*p20+p0*p22*p22*p25+p0*p25*p25*p22+p0*p18*p18*p24+p0*p24*p24*p18+p0*p19*p19*p24+p0*p24*p24*p19+p0*p21*p21*p23+p0*p23*p23*p21+p0*p20*p20*p22+p0*p22*p22*p20+p0*p18*p18*p19+p0*p19*p19*p18+p0*p15*p15*p17+p0*p17*p17*p15+p0*p16*p16*p17+p0*p17*p17*p16+p0*p15*p15*p16+p0*p16*p16*p15)/144;
     return;
  }


  if(b == 2){
     l=index(i,j,k,2); 
     double p0=mcL[l]; 
     l=index(i,j+1,k+1,0); 
     double p1=mcL[l]; 
     l=index(i+1,j,k+1,0); 
     double p2=mcL[l]; 
     l=index(i+1,j+1,k,0); 
     double p3=mcL[l]; 
     l=index(i+1,j+1,k+1,0); 
     double p4=mcL[l]; 
     l=index(i,j,k+1,1); 
     double p5=mcL[l]; 
     l=index(i,j+1,k,1); 
     double p6=mcL[l]; 
     l=index(i+1,j+1,k,1); 
     double p7=mcL[l]; 
     l=index(i+1,j,k+1,1); 
     double p8=mcL[l]; 
     l=index(i,j+1,k+1,1); 
     double p9=mcL[l]; 
     l=index(i+1,j,k,1); 
     double p10=mcL[l]; 
     l=index(i-1,j+1,k+1,1); 
     double p11=mcL[l]; 
     l=index(i+1,j+1,k+1,1); 
     double p12=mcL[l]; 
     l=index(i+1,j+1,k-1,1); 
     double p13=mcL[l]; 
     l=index(i+1,j-1,k+1,1); 
     double p14=mcL[l]; 
     l=index(i-1,j+1,k+1,0); 
     double p15=mcL[l]; 
     l=index(i-1,j+1,k+2,0); 
     double p16=mcL[l]; 
     l=index(i-1,j+2,k+1,0); 
     double p17=mcL[l]; 
     l=index(i+1,j-1,k+1,0); 
     double p18=mcL[l]; 
     l=index(i+1,j-1,k+2,0); 
     double p19=mcL[l]; 
     l=index(i+1,j+1,k-1,0); 
     double p20=mcL[l]; 
     l=index(i+1,j+1,k+2,0); 
     double p21=mcL[l]; 
     l=index(i+1,j+2,k-1,0); 
     double p22=mcL[l]; 
     l=index(i+1,j+2,k+1,0); 
     double p23=mcL[l]; 
     l=index(i+2,j-1,k+1,0); 
     double p24=mcL[l]; 
     l=index(i+2,j+1,k-1,0); 
     double p25=mcL[l]; 
     l=index(i+2,j+1,k+1,0); 
     double p26=mcL[l]; 
     l=index(i,j,k,0); 
     double p27=mcL[l]; 
     l=index(i,j,k+3,0); 
     double p28=mcL[l]; 
     l=index(i,j+3,k,0); 
     double p29=mcL[l]; 
     l=index(i+3,j,k,0); 
     double p30=mcL[l]; 
     l=index(i+2,j,k,2); 
     double p31=mcL[l]; 
     l=index(i-2,j,k,2); 
     double p32=mcL[l]; 
     l=index(i+2,j,k-2,2); 
     double p33=mcL[l]; 
     l=index(i-2,j,k+2,2); 
     double p34=mcL[l]; 
     l=index(i+2,j-2,k,2); 
     double p35=mcL[l]; 
     l=index(i-2,j+2,k,2); 
     double p36=mcL[l]; 
     l=index(i,j+2,k,2); 
     double p37=mcL[l]; 
     l=index(i,j-2,k,2); 
     double p38=mcL[l]; 
     l=index(i,j+2,k-2,2); 
     double p39=mcL[l]; 
     l=index(i,j-2,k+2,2); 
     double p40=mcL[l]; 
     l=index(i,j,k+2,2); 
     double p41=mcL[l]; 
     l=index(i,j,k-2,2); 
     double p42=mcL[l]; 
     l=index(i+2,j,k,0); 
     double p43=mcL[l]; 
     l=index(i+1,j,k,0); 
     double p44=mcL[l]; 
     l=index(i+2,j,k+1,0); 
     double p45=mcL[l]; 
     l=index(i+1,j,k+2,0); 
     double p46=mcL[l]; 
     l=index(i+2,j+1,k,0); 
     double p47=mcL[l]; 
     l=index(i+1,j+2,k,0); 
     double p48=mcL[l]; 
     l=index(i,j+2,k,0); 
     double p49=mcL[l]; 
     l=index(i,j+1,k,0); 
     double p50=mcL[l]; 
     l=index(i,j+2,k+1,0); 
     double p51=mcL[l]; 
     l=index(i,j+1,k+2,0); 
     double p52=mcL[l]; 
     l=index(i,j,k+2,0); 
     double p53=mcL[l]; 
     l=index(i,j,k+1,0); 
     double p54=mcL[l]; 
     l=index(i+2,j+2,k-1,0); 
     double p55=mcL[l]; 
     l=index(i+2,j-1,k+2,0); 
     double p56=mcL[l]; 
     l=index(i+2,j+2,k,0); 
     double p57=mcL[l]; 
     l=index(i+2,j-1,k,0); 
     double p58=mcL[l]; 
     l=index(i+2,j,k+2,0); 
     double p59=mcL[l]; 
     l=index(i+2,j,k-1,0); 
     double p60=mcL[l]; 
     l=index(i-1,j+2,k+2,0); 
     double p61=mcL[l]; 
     l=index(i-1,j+2,k,0); 
     double p62=mcL[l]; 
     l=index(i-1,j,k+2,0); 
     double p63=mcL[l]; 
     l=index(i,j+2,k+2,0); 
     double p64=mcL[l]; 
     l=index(i,j+2,k-1,0); 
     double p65=mcL[l]; 
     l=index(i,j-1,k+2,0); 
     double p66=mcL[l]; 

     AVcorr[0]+=1.0/3;
     AVcorr[1]+=(p0*p0)/2;
     AVcorr[2]+=(p0)/2;
     AVcorr[5]+=(p0*p0*p1+p0*p0*p2+p0*p0*p3+p0*p0*p4)/16;
     AVcorr[6]+=(p0*p1*p1+p0*p2*p2+p0*p3*p3+p0*p4*p4)/16;
     AVcorr[7]+=(p0*p0*p1*p1+p0*p0*p2*p2+p0*p0*p3*p3+p0*p0*p4*p4)/16;
     AVcorr[8]+=(p0*p1+p0*p2+p0*p3+p0*p4)/16;
     AVcorr[9]+=(p5*p5*p0+p0*p0*p5+p6*p6*p0+p0*p0*p6+p0*p0*p7+p7*p7*p0+p0*p0*p8+p8*p8*p0+p0*p0*p9+p9*p9*p0+p10*p10*p0+p0*p0*p10)/24;
     AVcorr[10]+=(p5*p5*p0*p0+p6*p6*p0*p0+p0*p0*p7*p7+p0*p0*p8*p8+p0*p0*p9*p9+p10*p10*p0*p0)/12;
     AVcorr[23]+=(p11*p11*p0*p0+p0*p0*p12*p12+p13*p13*p0*p0+p14*p14*p0*p0)/8;
     AVcorr[24]+=(p11*p0+p0*p12+p13*p0+p14*p0)/8;
     AVcorr[36]+=(p0*p0*p15*p15+p0*p0*p16*p16+p0*p0*p17*p17+p0*p0*p18*p18+p0*p0*p19*p19+p0*p0*p20*p20+p0*p0*p21*p21+p0*p0*p22*p22+p0*p0*p23*p23+p0*p0*p24*p24+p0*p0*p25*p25+p0*p0*p26*p26)/48;
     AVcorr[48]+=(p0*p0*p27+p0*p0*p28+p0*p0*p29+p0*p0*p30)/16;
     AVcorr[50]+=(p0*p0*p27*p27+p0*p0*p28*p28+p0*p0*p29*p29+p0*p0*p30*p30)/16;
     AVcorr[51]+=(p0*p27+p0*p28+p0*p29+p0*p30)/16;
     AVcorr[59]+=(p0*p0*p31+p32*p32*p0+p0*p0*p32+p31*p31*p0+p0*p0*p33+p34*p34*p0+p0*p0*p34+p33*p33*p0+p0*p0*p35+p36*p36*p0+p0*p0*p36+p35*p35*p0+p0*p0*p37+p38*p38*p0+p0*p0*p38+p37*p37*p0+p0*p0*p39+p40*p40*p0+p0*p0*p40+p39*p39*p0+p0*p0*p41+p42*p42*p0+p0*p0*p42+p41*p41*p0)/48;
     AVcorr[61]+=(p0*p31+p32*p0+p0*p33+p34*p0+p0*p35+p36*p0+p0*p37+p38*p0+p0*p39+p40*p0+p0*p41+p42*p0)/24;
     AVcorr[134]+=(p43*p44*p0*p0+p45*p46*p0*p0+p47*p48*p0*p0+p49*p50*p0*p0+p51*p52*p0*p0+p53*p54*p0*p0)/36;
     AVcorr[137]+=(p43*p44*p0+p45*p46*p0+p47*p48*p0+p49*p50*p0+p51*p52*p0+p53*p54*p0)/36;
     AVcorr[197]+=(p0*p4*p53*p53+p0*p4*p49*p49+p0*p3*p54*p54+p0*p3*p51*p51+p0*p2*p50*p50+p0*p2*p52*p52+p0*p4*p43*p43+p0*p3*p45*p45+p0*p2*p47*p47+p0*p1*p44*p44+p0*p1*p46*p46+p0*p1*p48*p48)/72;
     AVcorr[301]+=(p25*p25*p20*p0*p0+p20*p20*p25*p0*p0+p24*p24*p18*p0*p0+p18*p18*p24*p0*p0+p26*p26*p21*p0*p0+p21*p21*p26*p0*p0+p24*p24*p19*p0*p0+p19*p19*p24*p0*p0+p26*p26*p23*p0*p0+p23*p23*p26*p0*p0+p25*p25*p22*p0*p0+p22*p22*p25*p0*p0+p22*p22*p20*p0*p0+p20*p20*p22*p0*p0+p17*p17*p15*p0*p0+p15*p15*p17*p0*p0+p23*p23*p21*p0*p0+p21*p21*p23*p0*p0+p17*p17*p16*p0*p0+p16*p16*p17*p0*p0+p19*p19*p18*p0*p0+p18*p18*p19*p0*p0+p16*p16*p15*p0*p0+p15*p15*p16*p0*p0)/144;
     AVcorr[557]+=(p0*p47*p47*p24+p0*p45*p45*p25+p0*p43*p43*p26+p0*p48*p48*p17+p0*p51*p51*p22+p0*p49*p49*p23+p0*p46*p46*p16+p0*p52*p52*p19+p0*p44*p44*p15+p0*p50*p50*p18+p0*p53*p53*p21+p0*p54*p54*p20)/72;
     AVcorr[564]+=(p0*p0*p4*p16+p0*p0*p4*p17+p0*p0*p3*p15+p0*p0*p3*p17+p0*p0*p2*p15+p0*p0*p2*p16+p0*p0*p4*p19+p0*p0*p4*p24+p0*p0*p3*p18+p0*p0*p3*p24+p0*p0*p4*p22+p0*p0*p4*p25+p0*p0*p3*p23+p0*p0*p3*p26+p0*p0*p2*p20+p0*p0*p2*p25+p0*p0*p2*p21+p0*p0*p2*p26+p0*p0*p1*p18+p0*p0*p1*p19+p0*p0*p1*p20+p0*p0*p1*p22+p0*p0*p1*p21+p0*p0*p1*p23)/144;
     AVcorr[654]+=(p0*p0*p47*p14+p0*p0*p45*p13+p13*p13*p55*p0+p14*p14*p56*p0+p0*p0*p43*p12+p12*p12*p57*p0+p14*p14*p58*p0+p12*p12*p59*p0+p13*p13*p60*p0+p0*p0*p48*p11+p11*p11*p61*p0+p0*p0*p51*p13+p11*p11*p62*p0+p0*p0*p49*p12+p0*p0*p46*p11+p0*p0*p52*p14+p0*p0*p44*p11+p0*p0*p50*p14+p11*p11*p63*p0+p0*p0*p53*p12+p0*p0*p54*p13+p12*p12*p64*p0+p13*p13*p65*p0+p14*p14*p66*p0)/72;
     AVcorr[655]+=(p0*p47*p47*p14+p0*p45*p45*p13+p13*p55*p55*p0+p14*p56*p56*p0+p0*p43*p43*p12+p12*p57*p57*p0+p14*p58*p58*p0+p12*p59*p59*p0+p13*p60*p60*p0+p0*p48*p48*p11+p11*p61*p61*p0+p0*p51*p51*p13+p11*p62*p62*p0+p0*p49*p49*p12+p0*p46*p46*p11+p0*p52*p52*p14+p0*p44*p44*p11+p0*p50*p50*p14+p11*p63*p63*p0+p0*p53*p53*p12+p0*p54*p54*p13+p12*p64*p64*p0+p13*p65*p65*p0+p14*p66*p66*p0)/72;
     AVcorr[657]+=(p0*p47*p14*p14+p0*p45*p13*p13+p13*p55*p0*p0+p14*p56*p0*p0+p0*p43*p12*p12+p12*p57*p0*p0+p14*p58*p0*p0+p12*p59*p0*p0+p13*p60*p0*p0+p0*p48*p11*p11+p11*p61*p0*p0+p0*p51*p13*p13+p11*p62*p0*p0+p0*p49*p12*p12+p0*p46*p11*p11+p0*p52*p14*p14+p0*p44*p11*p11+p0*p50*p14*p14+p11*p63*p0*p0+p0*p53*p12*p12+p0*p54*p13*p13+p12*p64*p0*p0+p13*p65*p0*p0+p14*p66*p0*p0)/72;
     AVcorr[762]+=(p22*p22*p55*p0+p19*p19*p56*p0+p23*p23*p57*p0+p18*p18*p58*p0+p21*p21*p59*p0+p20*p20*p60*p0+p25*p25*p55*p0+p16*p16*p61*p0+p26*p26*p57*p0+p15*p15*p62*p0+p24*p24*p56*p0+p17*p17*p61*p0+p24*p24*p58*p0+p17*p17*p62*p0+p26*p26*p59*p0+p15*p15*p63*p0+p25*p25*p60*p0+p16*p16*p63*p0+p21*p21*p64*p0+p20*p20*p65*p0+p23*p23*p64*p0+p18*p18*p66*p0+p22*p22*p65*p0+p19*p19*p66*p0)/144;
     AVcorr[763]+=(p22*p55*p55*p0+p19*p56*p56*p0+p23*p57*p57*p0+p18*p58*p58*p0+p21*p59*p59*p0+p20*p60*p60*p0+p25*p55*p55*p0+p16*p61*p61*p0+p26*p57*p57*p0+p15*p62*p62*p0+p24*p56*p56*p0+p17*p61*p61*p0+p24*p58*p58*p0+p17*p62*p62*p0+p26*p59*p59*p0+p15*p63*p63*p0+p25*p60*p60*p0+p16*p63*p63*p0+p21*p64*p64*p0+p20*p65*p65*p0+p23*p64*p64*p0+p18*p66*p66*p0+p22*p65*p65*p0+p19*p66*p66*p0)/144;
     AVcorr[997]+=(p0*p24*p24*p26+p0*p26*p26*p24+p0*p25*p25*p26+p0*p26*p26*p25+p0*p24*p24*p25+p0*p25*p25*p24+p0*p17*p17*p23+p0*p23*p23*p17+p0*p22*p22*p23+p0*p23*p23*p22+p0*p17*p17*p22+p0*p22*p22*p17+p0*p16*p16*p21+p0*p21*p21*p16+p0*p19*p19*p21+p0*p21*p21*p19+p0*p15*p15*p20+p0*p20*p20*p15+p0*p18*p18*p20+p0*p20*p20*p18+p0*p16*p16*p19+p0*p19*p19*p16+p0*p15*p15*p18+p0*p18*p18*p15)/144;
     return;
  }


}
