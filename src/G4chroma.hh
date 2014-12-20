#ifndef __G4chroma_hh__
#define __G4chroma_hh__

#include <G4VModularPhysicsList.hh>
class ChromaPhysicsList: public G4VModularPhysicsList
{
public:
  ChromaPhysicsList();
  virtual ~ChromaPhysicsList();
  virtual void SetCuts();
};

#include <G4UserTrackingAction.hh>
#include <vector>
#include <G4ThreeVector.hh>

class PhotonTrackingAction : public G4UserTrackingAction
{
public:
  PhotonTrackingAction();
  virtual ~PhotonTrackingAction();
  
  int GetNumPhotons() const;
  void Clear();
  
  void GetX(double *x) const;
  void GetY(double *y) const;
  void GetZ(double *z) const;
  void GetDirX(double *dir_x) const;
  void GetDirY(double *dir_y) const;
  void GetDirZ(double *dir_z) const;
  void GetPolX(double *pol_x) const;
  void GetPolY(double *pol_y) const;
  void GetPolZ(double *pol_z) const;

  void GetWavelength(double *wl) const;
  void GetT0(double *t) const;

  // SCB testing setters 
  void AssignNumPhotons(int n);
  void SetX(const double *x);
  void SetY(const double *y);
  void SetZ(const double *z);
  void SetDirX(const double *dir_x);
  void SetDirY(const double *dir_y);
  void SetDirZ(const double *dir_z);


  virtual void PreUserTrackingAction(const G4Track *);

protected:
  std::vector<G4ThreeVector> pos;
  std::vector<G4ThreeVector> dir;
  std::vector<G4ThreeVector> pol;
  std::vector<double> wavelength;
  std::vector<double> t0;
};

#endif
